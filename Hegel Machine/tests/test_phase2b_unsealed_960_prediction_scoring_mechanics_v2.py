"""Synthetic tests for unsealed 960-row scoring mechanics V2.

Nothing in this file is an actual recognizer run, an authenticated answer reveal,
formal scoring, a formal gate evaluation, runtime or capacity evidence, an effect
estimate, or C1 evidence.  The only positive fixture is synthetic and unbacked.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
import ast
import hashlib
import inspect
from pathlib import Path
import runpy

import pytest

import hegel_machine.phase2b_formal_unsealed_prediction_scoring_contract_v2 as contract_v2
import hegel_machine.phase2b_recognizer_prediction_archive_v2 as archive_v2
import hegel_machine.phase2b_strict_recognizer_cli_v2 as strict_v2
import hegel_machine.phase2b_unsealed_960_prediction_scoring_mechanics_v2 as mechanics_v2
import hegel_machine.phase2b_unsealed_prediction_evaluator_v2 as evaluator_v2
from hegel_machine.phase2b_freeze_v1 import CanonicalFamilyId
from hegel_machine.phase2b_protocol import (
    Phase2BCaseType,
    salted_answer_commitment_sha256,
)
from hegel_machine.phase2b_recognizer_prediction_v2 import PredictionDecisionV2
from hegel_machine.phase2b_wire import RoleBinding


EXPECTED_MAIN_CASE_COUNTS = {
    Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE: 228,
    Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE: 12,
    Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE: 96,
    Phase2BCaseType.BINDING_COUNTERFACTUAL: 96,
    Phase2BCaseType.SCALE_COUNTERFACTUAL: 96,
    Phase2BCaseType.SIGN_OR_INVARIANT_BREAK: 96,
    Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE: 96,
}
EXPECTED_METRIC_DENOMINATORS = (
    ("answerable_count", 240),
    ("family_exact_accuracy", 240),
    ("binding_exact_accuracy", 240),
    ("scale_set_accuracy", 240),
    ("unique_scale_accuracy", 228),
    ("joint_exact_accuracy", 240),
    ("abstention_specificity", 228),
    ("nonidentifiability_abstention_accuracy", 96),
    ("set_valued_answer_accuracy", 12),
)
EXPECTED_METRIC_NAMES = tuple(name for name, _ in EXPECTED_METRIC_DENOMINATORS)
ANSWERABLE_CASE_TYPES = frozenset(
    {
        Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE,
        Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE,
    }
)
CONTROL_CASE_TYPES = frozenset(
    {
        Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE,
        Phase2BCaseType.BINDING_COUNTERFACTUAL,
        Phase2BCaseType.SCALE_COUNTERFACTUAL,
        Phase2BCaseType.SIGN_OR_INVARIANT_BREAK,
    }
)
EXPECTED_OVERALL_GATE_NAMES = (
    "family_exact",
    "binding_exact",
    "scale_set_accuracy",
    "joint_exact",
    "hard_negative_rejection",
    "binding_counterfactual_rejection",
    "scale_counterfactual_rejection",
    "sign_or_invariant_break_rejection",
    "abstention_specificity",
    "fail_closed_rate",
    "preservation_consistency",
    "nonidentifiable_scale_abstention",
)
EXPECTED_SLICE_GATE_REFERENCES = (
    ("answerable_joint_exact", "family"),
    ("all_control_rejection", "family"),
    ("abstention_specificity", "family"),
    ("answerable_joint_exact", "scale"),
    ("all_control_rejection", "scale"),
    ("abstention_specificity", "scale"),
)
PUBLIC_TYPES = (
    "Unsealed960MetricRowOutcomeV2",
    "Unsealed960MainRowResultV2",
    "Unsealed960MetricResultV2",
    "Unsealed960PredictionScoringMechanicsV2",
    "Unsealed960PredictionScoringRejectionV2",
)
PUBLIC_ENUMS = (
    "Unsealed960PredictionScoringDispositionV2",
    "Unsealed960PredictionScoringReasonV2",
)
METRIC_ROW_OUTCOME_FIELDS = (
    "metric_definition_id",
    "metric_name",
    "eligible",
    "success",
    "metric_row_outcome_id",
)
MAIN_ROW_RESULT_FIELDS = (
    "input_row_id",
    "prediction_record_id",
    "prediction_content_id",
    "answer_row_id",
    "case_type",
    "predicted_decision",
    "expected_decision",
    "predicted_canonical_family_id",
    "expected_canonical_family_id",
    "predicted_binding",
    "expected_binding",
    "predicted_admissible_scale_ids",
    "expected_admissible_scale_ids",
    "decision_exact",
    "family_exact",
    "binding_exact",
    "scale_set_exact",
    "joint_exact",
    "metric_eligible",
    "metric_outcomes",
    "row_result_id",
)
METRIC_RESULT_FIELDS = (
    "metric_definition_id",
    "metric_name",
    "metric_kind",
    "denominator_case_types",
    "expected_denominator",
    "observed_denominator",
    "success_count",
    "count_value",
    "success_rule",
    "separately_reported",
    "metric_result_id",
)
TRUE_RESULT_CLAIMS = (
    "canonical_prediction_archive_replay_verified",
    "formal_contract_validation_replayed",
    "supplied_answer_commitment_opening_verified",
    "prediction_archive_context_cross_binding_verified",
    "exact_main_720_row_join_verified",
    "semantic_conflict_240_excluded_from_metrics",
    "nine_metric_results_materialized",
    "exact_720_main_row_results_materialized",
    "supplied_archive_nine_metric_mechanics_performed",
)
FALSE_RESULT_CLAIMS = (
    "challenge_in_main_denominator",
    "challenge_scoring_performed",
    "control_rejection_metrics_implemented",
    "formal_gate_evaluation_performed",
    "overall_gate_results_materialized",
    "slice_gate_metrics_implemented",
    "scale_regret_evaluated",
    "bootstrap_evaluated",
    "answer_manifest_authority_verified",
    "answer_commitment_authority_verified",
    "pre_reveal_commitment_timing_verified",
    "input_archive_membership_verified",
    "batch_policy_membership_verified",
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
RESULT_FIELDS = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "result_id",
    "prediction_archive_id",
    "prediction_archive_sha256",
    "prediction_archive_version",
    "prediction_archive_policy_id",
    "run_context_id",
    "input_archive_id",
    "input_archive_sha256",
    "batch_id",
    "execution_freeze_manifest_id",
    "protocol_id",
    "structural_receipt_id",
    "partition_manifest_id",
    "answer_manifest_id",
    "answer_manifest_sha256",
    "salted_answer_commitment_sha256",
    "formal_scoring_contract_id",
    "ordered_archive_input_row_ids_root",
    "main_row_ids_root",
    "semantic_conflict_row_ids_root",
    "partition_union_row_ids_root",
    "main_answer_row_ids_root",
    "total_prediction_count",
    "main_row_result_count",
    "metric_eligible_main_row_count",
    "control_row_without_frozen_metric_count",
    "semantic_conflict_excluded_count",
    *TRUE_RESULT_CLAIMS,
    *FALSE_RESULT_CLAIMS,
    "metric_results",
    "main_row_results",
    "gate_results",
    "scale_regret_result",
    "bootstrap_result",
)
REJECTION_FIELDS = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "result",
    "metric_results",
    "main_row_results",
    "gate_results",
    "scale_regret_result",
    "bootstrap_result",
    "partial_output_published",
    *TRUE_RESULT_CLAIMS,
    *FALSE_RESULT_CLAIMS,
)
FORBIDDEN_MODULE_SUFFIXES = frozenset(
    {
        "phase2b_recognizer_prediction_archive_v1",
        "phase2b_unsealed_prediction_evaluator_v1",
        "phase2b_runner",
        "phase2b_selector",
    }
)
FORBIDDEN_CALL_TERMINALS = frozenset(
    {
        "build_recognizer_prediction_archive_v2",
        "build_unsealed_prediction_partition_manifest_v2",
        "evaluate_binary_gate",
        "evaluate_unsealed_prediction_archive_structure_v2",
        "one_sided_wilson_lower_bound",
        "paired_cluster_bootstrap",
        "recognize_public_input_row_v2",
        "run_recognizer",
        "score_prediction",
        "score_predictions",
    }
)


class _ForbiddenBoundaryReached(BaseException):
    """Sentinel proving malformed input reached replay, hashing, or scoring."""


def _field_names(value_type: type[object]) -> tuple[str, ...]:
    return tuple(item.name for item in fields(value_type))


def _unchecked_copy(value: object, **changes: object) -> object:
    copied = object.__new__(type(value))
    for item in fields(value):
        object.__setattr__(
            copied,
            item.name,
            changes.get(item.name, object.__getattribute__(value, item.name)),
        )
    return copied


def _ast_direct_imports_and_calls(
    source: str,
) -> tuple[frozenset[str], frozenset[str]]:
    tree = ast.parse(source)
    aliases: dict[str, str] = {}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                imported.add(item.name)
                local = item.asname or item.name.split(".", 1)[0]
                aliases[local] = item.name if item.asname else local
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.add(module)
            for item in node.names:
                if item.name == "*":
                    imported.add(f"{module}.*".strip("."))
                    continue
                target = f"{module}.{item.name}".strip(".")
                imported.add(target)
                aliases[item.asname or item.name] = target

    def qualified_name(value: ast.expr) -> str | None:
        if isinstance(value, ast.Name):
            return aliases.get(value.id, value.id)
        if isinstance(value, ast.Attribute):
            parent = qualified_name(value.value)
            return value.attr if parent is None else f"{parent}.{value.attr}"
        return None

    calls = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in (qualified_name(node.func),)
        if name is not None
    }
    return frozenset(imported), frozenset(calls)


@dataclass(frozen=True, slots=True)
class _MixedScoringFixtureV2:
    prediction_archive_bytes: bytes
    decoded_prediction_archive: archive_v2.DecodedRecognizerPredictionArchiveV2
    structural_receipt: strict_v2.StrictRecognizerStructuralReceiptV2
    structural_evaluation: evaluator_v2.UnsealedPredictionStructuralEvaluationV2
    partition_manifest: evaluator_v2.UnsealedPredictionPartitionManifestV2
    answer_manifest: contract_v2.FormalUnsealedAnswerManifestV2
    answer_commitment_salt: str
    salted_answer_commitment_sha256: str
    expected_metrics: tuple[tuple[str, int, int], ...]


def _uuid4(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012x}"


def _different_family(value: CanonicalFamilyId) -> CanonicalFamilyId:
    return next(item for item in CanonicalFamilyId if item is not value)


def _case_type_sequence() -> tuple[Phase2BCaseType, ...]:
    value = tuple(
        case_type
        for case_type in Phase2BCaseType
        for _ in range(EXPECTED_MAIN_CASE_COUNTS[case_type])
    )
    assert len(value) == 720
    return value


def _metric_oracle(
    *,
    answer_rows: tuple[contract_v2.FormalUnsealedAnswerRowV2, ...],
    records: tuple[archive_v2.PublicRecognizerPredictionRecordV2, ...],
) -> tuple[tuple[str, int, int], ...]:
    """Independent, integer-only oracle for the frozen nine metric rules."""

    by_id = {record.input_row_id: record for record in records}
    assert len(by_id) == 960
    numerators = {name: 0 for name in EXPECTED_METRIC_NAMES}
    denominators = {name: 0 for name in EXPECTED_METRIC_NAMES}
    for answer in answer_rows:
        record = by_id[answer.input_row_id]
        decision_positive = record.decision in {
            PredictionDecisionV2.ANSWER,
            PredictionDecisionV2.ANSWER_SET,
        }
        family_exact = (
            decision_positive
            and record.canonical_family_id is answer.canonical_family_id
        )
        binding_exact = (
            decision_positive and record.prediction.binding == answer.binding
        )
        scale_exact = (
            decision_positive
            and record.prediction.admissible_scale_ids
            == answer.admissible_scale_ids
        )
        if answer.case_type in ANSWERABLE_CASE_TYPES:
            for name in (
                "answerable_count",
                "family_exact_accuracy",
                "binding_exact_accuracy",
                "scale_set_accuracy",
                "joint_exact_accuracy",
            ):
                denominators[name] += 1
            numerators["answerable_count"] += 1
            numerators["family_exact_accuracy"] += int(family_exact)
            numerators["binding_exact_accuracy"] += int(binding_exact)
            numerators["scale_set_accuracy"] += int(scale_exact)
            numerators["joint_exact_accuracy"] += int(
                record.decision is answer.expected_decision
                and family_exact
                and binding_exact
                and scale_exact
            )
        if answer.case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE:
            denominators["unique_scale_accuracy"] += 1
            denominators["abstention_specificity"] += 1
            numerators["unique_scale_accuracy"] += int(
                record.decision is PredictionDecisionV2.ANSWER and scale_exact
            )
            numerators["abstention_specificity"] += int(
                record.decision is not PredictionDecisionV2.ABSTAIN
            )
        elif answer.case_type is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE:
            denominators["set_valued_answer_accuracy"] += 1
            numerators["set_valued_answer_accuracy"] += int(
                record.decision is PredictionDecisionV2.ANSWER_SET
                and family_exact
                and binding_exact
                and scale_exact
            )
        elif (
            answer.case_type
            is Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE
        ):
            denominators["nonidentifiability_abstention_accuracy"] += 1
            numerators["nonidentifiability_abstention_accuracy"] += int(
                record.decision is PredictionDecisionV2.ABSTAIN
            )
    assert tuple(denominators.items()) == EXPECTED_METRIC_DENOMINATORS
    return tuple(
        (name, numerators[name], denominator)
        for name, denominator in EXPECTED_METRIC_DENOMINATORS
    )


@pytest.fixture(scope="module")
def mixed_scoring_fixture_v2() -> _MixedScoringFixtureV2:
    """Build one mixed unbacked archive and all upstream structural objects."""

    archive_namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_recognizer_prediction_archive_v2.py"))
    )
    freeze = archive_namespace["execution_freeze_manifest"].__wrapped__()
    base = archive_namespace["synthetic_archive"].__wrapped__(freeze)
    base_decoded = base.decoded
    main_row_ids = tuple(sorted(base_decoded.input_row_ids[:720]))
    conflict_row_ids = tuple(sorted(base_decoded.input_row_ids[720:]))
    case_by_id = dict(zip(main_row_ids, _case_type_sequence(), strict=True))
    ordinal_by_case: dict[Phase2BCaseType, int] = {
        case_type: 0 for case_type in Phase2BCaseType
    }
    root_indices = (1, 0, *range(2, 960))
    records: list[archive_v2.PublicRecognizerPredictionRecordV2] = []
    for wire_index, (base_record, root_index) in enumerate(
        zip(base_decoded.records, root_indices, strict=True)
    ):
        case_type = case_by_id.get(base_record.input_row_id)
        if case_type is None:
            make_positive = wire_index % 2 == 0
            decision = (
                PredictionDecisionV2.ANSWER
                if wire_index % 4 == 0
                else PredictionDecisionV2.ANSWER_SET
            )
        else:
            ordinal = ordinal_by_case[case_type]
            ordinal_by_case[case_type] += 1
            if case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE:
                make_positive = ordinal % 5 != 4
                decision = PredictionDecisionV2.ANSWER
            elif case_type is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE:
                make_positive = ordinal % 5 != 4
                decision = PredictionDecisionV2.ANSWER_SET
            elif case_type is Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE:
                make_positive = ordinal % 4 == 0
                decision = PredictionDecisionV2.ANSWER
            else:
                make_positive = ordinal % 3 == 0
                decision = PredictionDecisionV2.ANSWER
        if not make_positive:
            records.append(base_record)
            continue
        root_row = archive_namespace["_synthetic_root_row"](root_index)
        outcome = archive_namespace["_positive_outcome"](
            row_id=root_row.row_id,
            payload_sha256=root_row.payload_sha256,
            freeze_manifest_id=base_decoded.context.execution_freeze_manifest_id,
            index=10_000 + wire_index,
            decision=decision,
        )
        records.append(
            archive_v2.PublicRecognizerPredictionRecordV2._issue(
                archive_v2._RECORD_ISSUE_TOKEN_V2,
                context=base_decoded.context,
                input_row=root_row,
                outcome=outcome,
            )
        )
    raw_archive = archive_v2._encode_prediction_archive_v2(
        context=base_decoded.context,
        records=tuple(records),
    )
    decoded = archive_v2.decode_public_recognizer_prediction_archive_v2(raw_archive)
    assert decoded.input_row_ids == base_decoded.input_row_ids
    record_by_id = {record.input_row_id: record for record in decoded.records}
    answer_rows: list[contract_v2.FormalUnsealedAnswerRowV2] = []
    ordinal_by_case = {case_type: 0 for case_type in Phase2BCaseType}
    families = tuple(CanonicalFamilyId)
    for answer_index, (input_row_id, case_type) in enumerate(
        zip(main_row_ids, _case_type_sequence(), strict=True)
    ):
        record = record_by_id[input_row_id]
        ordinal = ordinal_by_case[case_type]
        ordinal_by_case[case_type] += 1
        if case_type in ANSWERABLE_CASE_TYPES:
            expected_decision = (
                PredictionDecisionV2.ANSWER
                if case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE
                else PredictionDecisionV2.ANSWER_SET
            )
            if record.decision is expected_decision:
                expected_family = record.canonical_family_id
                expected_binding = record.prediction.binding
                expected_scales = record.prediction.admissible_scale_ids
            else:
                expected_family = families[0]
                expected_binding = (
                    RoleBinding(
                        role_id=_uuid4(800_000 + answer_index),
                        entity_id=_uuid4(900_000 + answer_index),
                    ),
                )
                scale_count = 1 if expected_decision is PredictionDecisionV2.ANSWER else 2
                expected_scales = tuple(
                    _uuid4(1_000_000 + 4 * answer_index + offset)
                    for offset in range(scale_count)
                )
            mismatch = ordinal % 5
            if record.decision is expected_decision and mismatch == 1:
                expected_family = _different_family(expected_family)
            elif record.decision is expected_decision and mismatch == 2:
                expected_binding = (
                    RoleBinding(
                        role_id=_uuid4(1_100_000 + answer_index),
                        entity_id=_uuid4(1_200_000 + answer_index),
                    ),
                )
            elif record.decision is expected_decision and mismatch == 3:
                expected_scales = tuple(
                    _uuid4(1_300_000 + 4 * answer_index + offset)
                    for offset in range(len(expected_scales))
                )
        else:
            expected_decision = PredictionDecisionV2.ABSTAIN
            expected_family = None
            expected_binding = ()
            expected_scales = ()
        answer_rows.append(
            contract_v2.FormalUnsealedAnswerRowV2(
                input_row_id=input_row_id,
                case_type=case_type,
                expected_decision=expected_decision,
                canonical_family_id=expected_family,
                binding=expected_binding,
                admissible_scale_ids=expected_scales,
            )
        )
    partition = evaluator_v2.build_unsealed_prediction_partition_manifest_v2(
        prediction_archive=decoded,
        main_row_ids=main_row_ids,
        semantic_conflict_row_ids=conflict_row_ids,
    )
    evaluation = evaluator_v2.evaluate_unsealed_prediction_archive_structure_v2(
        prediction_archive=decoded,
        partition_manifest=partition,
    )
    assert type(evaluation) is evaluator_v2.UnsealedPredictionStructuralEvaluationV2
    context = decoded.context
    receipt = strict_v2.StrictRecognizerStructuralReceiptV2._issue(
        strict_v2._RECEIPT_ISSUE_TOKEN_V2,
        input_archive_id=context.input_archive_id,
        input_archive_sha256=context.input_archive_sha256,
        input_archive_version=strict_v2.TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
        input_archive_policy_id=strict_v2.RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
        prediction_archive_id=decoded.archive_id,
        prediction_archive_sha256=hashlib.sha256(raw_archive).hexdigest(),
        prediction_archive_version=decoded.schema_version,
        prediction_archive_policy_id=decoded.policy_id,
        batch_id=context.batch_id,
        batch_policy_id=context.batch_policy_id,
        run_context_id=context.context_id,
        execution_freeze_manifest_id=context.execution_freeze_manifest_id,
        protocol_id=context.protocol_id,
    )
    answer = contract_v2.build_formal_unsealed_answer_manifest_v2(
        input_archive_id=receipt.input_archive_id,
        input_archive_sha256=receipt.input_archive_sha256,
        input_archive_version=receipt.input_archive_version,
        input_archive_policy_id=receipt.input_archive_policy_id,
        batch_id=receipt.batch_id,
        batch_policy_id=receipt.batch_policy_id,
        exact_freeze_id=partition.exact_freeze_id,
        phase2b_protocol_id=receipt.protocol_id,
        execution_freeze_manifest_id=receipt.execution_freeze_manifest_id,
        ordered_archive_input_row_ids_root=partition.ordered_archive_input_row_ids_root,
        main_row_ids_root=partition.main_row_ids_root,
        semantic_conflict_row_ids_root=partition.semantic_conflict_row_ids_root,
        partition_union_row_ids_root=partition.partition_union_row_ids_root,
        main_answer_rows=tuple(answer_rows),
    )
    salt = "synthetic-unsealed-scoring-mechanics-only-salt-0123456789abcdef"
    commitment = salted_answer_commitment_sha256(
        answer.answer_manifest_sha256,
        salt,
    )
    expected_metrics = _metric_oracle(
        answer_rows=answer.main_answer_rows,
        records=decoded.records,
    )
    return _MixedScoringFixtureV2(
        prediction_archive_bytes=raw_archive,
        decoded_prediction_archive=decoded,
        structural_receipt=receipt,
        structural_evaluation=evaluation,
        partition_manifest=partition,
        answer_manifest=answer,
        answer_commitment_salt=salt,
        salted_answer_commitment_sha256=commitment,
        expected_metrics=expected_metrics,
    )


def test_synthetic_fixture_is_mixed_unbacked_and_has_independent_oracle(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
) -> None:
    fixture = mixed_scoring_fixture_v2
    decisions = tuple(
        record.decision for record in fixture.decoded_prediction_archive.records
    )
    assert set(decisions) == set(PredictionDecisionV2)
    assert tuple(len(group) for group in (
        fixture.partition_manifest.main_row_ids,
        fixture.partition_manifest.semantic_conflict_row_ids,
    )) == (720, 240)
    assert len(fixture.answer_manifest.main_answer_rows) == 720
    assert fixture.decoded_prediction_archive.input_row_ids != tuple(
        sorted(fixture.decoded_prediction_archive.input_row_ids)
    )
    assert fixture.expected_metrics == (
        ("answerable_count", 240, 240),
        ("family_exact_accuracy", 154, 240),
        ("binding_exact_accuracy", 154, 240),
        ("scale_set_accuracy", 152, 240),
        ("unique_scale_accuracy", 144, 228),
        ("joint_exact_accuracy", 74, 240),
        ("abstention_specificity", 183, 228),
        ("nonidentifiability_abstention_accuracy", 72, 96),
        ("set_valued_answer_accuracy", 3, 12),
    )


def _score_kwargs(
    fixture: _MixedScoringFixtureV2,
    **changes: object,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "prediction_archive": fixture.prediction_archive_bytes,
        "structural_receipt": fixture.structural_receipt,
        "structural_evaluation": fixture.structural_evaluation,
        "partition_manifest": fixture.partition_manifest,
        "answer_manifest": fixture.answer_manifest,
        "revealed_answer_manifest_sha256": (
            fixture.answer_manifest.answer_manifest_sha256
        ),
        "answer_commitment_salt": fixture.answer_commitment_salt,
        "salted_answer_commitment_sha256": (
            fixture.salted_answer_commitment_sha256
        ),
    }
    kwargs.update(changes)
    return kwargs


def _fixture_with_replaced_challenge_records(
    fixture: _MixedScoringFixtureV2,
) -> _MixedScoringFixtureV2:
    """Change all 240 challenge predictions while retaining row IDs and main rows."""

    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_recognizer_prediction_archive_v2.py"))
    )

    challenge_ids = set(fixture.partition_manifest.semantic_conflict_row_ids)
    root_indices = (1, 0, *range(2, 960))
    records: list[archive_v2.PublicRecognizerPredictionRecordV2] = []
    for wire_index, (record, root_index) in enumerate(
        zip(
            fixture.decoded_prediction_archive.records,
            root_indices,
            strict=True,
        )
    ):
        if record.input_row_id not in challenge_ids:
            records.append(record)
            continue
        root_row = namespace["_synthetic_root_row"](root_index)
        replacement = namespace["_outcome"](
            row_id=root_row.row_id,
            payload_sha256=root_row.payload_sha256,
            freeze_manifest_id=(
                fixture.decoded_prediction_archive.context
                .execution_freeze_manifest_id
            ),
            index=50_000 + wire_index,
        )
        records.append(
            archive_v2.PublicRecognizerPredictionRecordV2._issue(
                archive_v2._RECORD_ISSUE_TOKEN_V2,
                context=fixture.decoded_prediction_archive.context,
                input_row=root_row,
                outcome=replacement,
            )
        )
    raw = archive_v2._encode_prediction_archive_v2(
        context=fixture.decoded_prediction_archive.context,
        records=tuple(records),
    )
    assert raw != fixture.prediction_archive_bytes
    decoded = archive_v2.decode_public_recognizer_prediction_archive_v2(raw)
    partition = evaluator_v2.build_unsealed_prediction_partition_manifest_v2(
        prediction_archive=decoded,
        main_row_ids=fixture.partition_manifest.main_row_ids,
        semantic_conflict_row_ids=(
            fixture.partition_manifest.semantic_conflict_row_ids
        ),
    )
    evaluation = evaluator_v2.evaluate_unsealed_prediction_archive_structure_v2(
        prediction_archive=decoded,
        partition_manifest=partition,
    )
    assert type(evaluation) is evaluator_v2.UnsealedPredictionStructuralEvaluationV2
    context = decoded.context
    receipt = strict_v2.StrictRecognizerStructuralReceiptV2._issue(
        strict_v2._RECEIPT_ISSUE_TOKEN_V2,
        input_archive_id=context.input_archive_id,
        input_archive_sha256=context.input_archive_sha256,
        input_archive_version=strict_v2.TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
        input_archive_policy_id=strict_v2.RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
        prediction_archive_id=decoded.archive_id,
        prediction_archive_sha256=hashlib.sha256(raw).hexdigest(),
        prediction_archive_version=decoded.schema_version,
        prediction_archive_policy_id=decoded.policy_id,
        batch_id=context.batch_id,
        batch_policy_id=context.batch_policy_id,
        run_context_id=context.context_id,
        execution_freeze_manifest_id=context.execution_freeze_manifest_id,
        protocol_id=context.protocol_id,
    )
    return _MixedScoringFixtureV2(
        prediction_archive_bytes=raw,
        decoded_prediction_archive=decoded,
        structural_receipt=receipt,
        structural_evaluation=evaluation,
        partition_manifest=partition,
        answer_manifest=fixture.answer_manifest,
        answer_commitment_salt=fixture.answer_commitment_salt,
        salted_answer_commitment_sha256=(
            fixture.salted_answer_commitment_sha256
        ),
        expected_metrics=fixture.expected_metrics,
    )


@pytest.fixture(scope="module")
def challenge_replaced_fixture_v2(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
) -> _MixedScoringFixtureV2:
    return _fixture_with_replaced_challenge_records(mixed_scoring_fixture_v2)


def _assert_atomic_rejection(
    value: object,
    reason: mechanics_v2.Unsealed960PredictionScoringReasonV2,
) -> None:
    assert type(value) is mechanics_v2.Unsealed960PredictionScoringRejectionV2
    assert value.disposition is (
        mechanics_v2.Unsealed960PredictionScoringDispositionV2.REJECTED
    )
    assert value.reason is reason
    assert value.result is None
    assert value.metric_results == ()
    assert value.main_row_results == ()
    assert value.gate_results == ()
    assert value.scale_regret_result is None
    assert value.bootstrap_result is None
    assert value.partial_output_published is False
    for name in (*TRUE_RESULT_CLAIMS, *FALSE_RESULT_CLAIMS):
        assert type(getattr(value, name)) is bool
        assert getattr(value, name) is False


def test_public_surface_signature_and_field_manifests_are_exact() -> None:
    assert mechanics_v2.__all__ == (
        "UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_VERSION",
        "UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_CLAIM_LEVEL",
        "UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_SCHEMA_ID",
        "UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_POLICY_ID",
        *PUBLIC_ENUMS,
        *PUBLIC_TYPES,
        "score_unsealed_960_prediction_scoring_mechanics_v2",
    )
    signature = inspect.signature(
        mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2
    )
    assert tuple(signature.parameters) == (
        "prediction_archive",
        "structural_receipt",
        "structural_evaluation",
        "partition_manifest",
        "answer_manifest",
        "revealed_answer_manifest_sha256",
        "answer_commitment_salt",
        "salted_answer_commitment_sha256",
    )
    assert all(
        item.kind is inspect.Parameter.KEYWORD_ONLY
        for item in signature.parameters.values()
    )
    assert _field_names(mechanics_v2.Unsealed960MetricRowOutcomeV2) == (
        METRIC_ROW_OUTCOME_FIELDS
    )
    assert _field_names(mechanics_v2.Unsealed960MainRowResultV2) == (
        MAIN_ROW_RESULT_FIELDS
    )
    assert _field_names(mechanics_v2.Unsealed960MetricResultV2) == (
        METRIC_RESULT_FIELDS
    )
    assert _field_names(mechanics_v2.Unsealed960PredictionScoringMechanicsV2) == (
        RESULT_FIELDS
    )
    assert _field_names(mechanics_v2.Unsealed960PredictionScoringRejectionV2) == (
        REJECTION_FIELDS
    )
    assert mechanics_v2._TRUE_RESULT_CLAIMS_V2 == TRUE_RESULT_CLAIMS
    assert mechanics_v2._FALSE_RESULT_CLAIMS_V2 == FALSE_RESULT_CLAIMS
    for value_type in (
        mechanics_v2.Unsealed960MetricRowOutcomeV2,
        mechanics_v2.Unsealed960MainRowResultV2,
        mechanics_v2.Unsealed960MetricResultV2,
        mechanics_v2.Unsealed960PredictionScoringMechanicsV2,
        mechanics_v2.Unsealed960PredictionScoringRejectionV2,
    ):
        with pytest.raises(TypeError, match="privately issued"):
            value_type()


def test_public_identity_and_closed_disposition_reason_literals() -> None:
    assert mechanics_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_VERSION == (
        "hegel-machine-phase2b-unsealed-960-prediction-scoring-mechanics/2"
    )
    assert mechanics_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_CLAIM_LEVEL == (
        "NON_AUTHORITATIVE_SCORING_MECHANICS_ONLY"
    )
    assert mechanics_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_SCHEMA_ID == (
        "phase2b_unsealed_960_prediction_scoring_mechanics_schema_v2_"
        "b43a50271b4ee645daa9a33f80ac45bd7e3ed0b59d237a4ff7c1fa4a5b2997ed"
    )
    assert mechanics_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_POLICY_ID == (
        "phase2b_unsealed_960_prediction_scoring_mechanics_policy_v2_"
        "196ebbb1f4c1e1c3d5e13ab4da2d490cc7e8e0d5d173093cd2caa674f1acdc1b"
    )
    assert tuple(
        item.value
        for item in mechanics_v2.Unsealed960PredictionScoringDispositionV2
    ) == (
        "MECHANICS_COMPLETE_NOT_ACTUAL_EXECUTION",
        "REJECTED",
    )
    assert tuple(
        item.value for item in mechanics_v2.Unsealed960PredictionScoringReasonV2
    ) == (
        "CANONICAL_V2_MAIN_ROW_NINE_METRIC_MECHANICS_COMPLETE",
        "WRONG_INPUT_TYPE",
        "FORMAL_CONTRACT_REJECTED",
        "PREDICTION_ARCHIVE_INVALID",
        "PREDICTION_ARCHIVE_BINDING_MISMATCH",
        "MAIN_ROW_JOIN_MISMATCH",
        "METRIC_DENOMINATOR_MISMATCH",
        "INTERNAL_ERROR",
    )


def test_source_has_no_v1_private_runner_mapper_gate_or_bootstrap_calls() -> None:
    source = Path(mechanics_v2.__file__).read_text(encoding="utf-8")
    imported, calls = _ast_direct_imports_and_calls(source)
    assert not {
        name
        for name in imported
        if name.rsplit(".", 1)[-1] in FORBIDDEN_MODULE_SUFFIXES
    }
    forbidden_calls = {
        call
        for call in calls
        if call.rsplit(".", 1)[-1] in FORBIDDEN_CALL_TERMINALS
        or call.rsplit(".", 1)[-1].startswith(
            ("build_recognizer_", "evaluate_unsealed_", "run_recognizer")
        )
        or "bootstrap" in call.rsplit(".", 1)[-1].casefold()
        or "gate" in call.rsplit(".", 1)[-1].casefold()
    }
    assert forbidden_calls == set()


@pytest.mark.parametrize(
    "field_name",
    (
        "structural_receipt",
        "structural_evaluation",
        "partition_manifest",
        "answer_manifest",
    ),
)
def test_wrong_wrapper_type_is_atomic_without_partial_output(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
    field_name: str,
) -> None:
    result = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(mixed_scoring_fixture_v2, **{field_name: object()})
    )
    _assert_atomic_rejection(
        result,
        mechanics_v2.Unsealed960PredictionScoringReasonV2.WRONG_INPUT_TYPE,
    )


@pytest.mark.parametrize(
    "prediction_archive",
    (
        bytearray(b"x"),
        memoryview(b"x"),
        b"",
        b"x" * (archive_v2.MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2 + 1),
    ),
)
def test_raw_prediction_bytes_exact_type_and_cap_reject_atomically(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
    prediction_archive: object,
) -> None:
    result = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(
            mixed_scoring_fixture_v2,
            prediction_archive=prediction_archive,
        )
    )
    expected = (
        mechanics_v2.Unsealed960PredictionScoringReasonV2.WRONG_INPUT_TYPE
        if type(prediction_archive) is not bytes
        else mechanics_v2.Unsealed960PredictionScoringReasonV2.PREDICTION_ARCHIVE_INVALID
    )
    _assert_atomic_rejection(result, expected)


def test_raw_byte_shape_gate_precedes_validator_decoder_and_sha256(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise _ForbiddenBoundaryReached("malformed bytes reached deep work")

    monkeypatch.setattr(
        mechanics_v2,
        "validate_formal_unsealed_prediction_scoring_contract_v2",
        forbidden,
    )
    monkeypatch.setattr(
        mechanics_v2,
        "decode_public_recognizer_prediction_archive_v2",
        forbidden,
    )
    monkeypatch.setattr(mechanics_v2.hashlib, "sha256", forbidden)
    result = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(mixed_scoring_fixture_v2, prediction_archive=b"")
    )
    _assert_atomic_rejection(
        result,
        mechanics_v2.Unsealed960PredictionScoringReasonV2
        .PREDICTION_ARCHIVE_INVALID,
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("revealed_answer_manifest_sha256", object()),
        ("answer_commitment_salt", object()),
        ("salted_answer_commitment_sha256", object()),
    ),
)
def test_late_opening_scalar_global_preflight_precedes_public_calls(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    invalid_value: object,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise _ForbiddenBoundaryReached("malformed opening reached public replay")

    monkeypatch.setattr(
        mechanics_v2,
        "validate_formal_unsealed_prediction_scoring_contract_v2",
        forbidden,
    )
    monkeypatch.setattr(
        mechanics_v2,
        "decode_public_recognizer_prediction_archive_v2",
        forbidden,
    )
    result = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(
            mixed_scoring_fixture_v2,
            **{field_name: invalid_value},
        )
    )
    _assert_atomic_rejection(
        result,
        mechanics_v2.Unsealed960PredictionScoringReasonV2.WRONG_INPUT_TYPE,
    )


@pytest.mark.parametrize("row_index", (0, 360, 719))
def test_first_middle_last_malformed_answer_row_is_atomic_before_decode(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
    monkeypatch: pytest.MonkeyPatch,
    row_index: int,
) -> None:
    rows = list(mixed_scoring_fixture_v2.answer_manifest.main_answer_rows)
    rows[row_index] = _unchecked_copy(rows[row_index], input_row_id="malformed")
    malformed = _unchecked_copy(
        mixed_scoring_fixture_v2.answer_manifest,
        main_answer_rows=tuple(rows),
    )

    def forbidden_decode(*args: object, **kwargs: object) -> object:
        raise _ForbiddenBoundaryReached(
            "malformed answer row reached prediction decode"
        )

    monkeypatch.setattr(
        mechanics_v2,
        "decode_public_recognizer_prediction_archive_v2",
        forbidden_decode,
    )
    result = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(
            mixed_scoring_fixture_v2,
            answer_manifest=malformed,
        )
    )
    _assert_atomic_rejection(
        result,
        mechanics_v2.Unsealed960PredictionScoringReasonV2
        .FORMAL_CONTRACT_REJECTED,
    )


def test_upstream_false_claim_pollution_cannot_authorize_scoring(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polluted = _unchecked_copy(
        mixed_scoring_fixture_v2.structural_receipt,
        actual_960_case_run_verified=True,
    )

    def forbidden_decode(*args: object, **kwargs: object) -> object:
        raise _ForbiddenBoundaryReached("polluted receipt reached archive decode")

    monkeypatch.setattr(
        mechanics_v2,
        "decode_public_recognizer_prediction_archive_v2",
        forbidden_decode,
    )
    result = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(
            mixed_scoring_fixture_v2,
            structural_receipt=polluted,
        )
    )
    _assert_atomic_rejection(
        result,
        mechanics_v2.Unsealed960PredictionScoringReasonV2
        .FORMAL_CONTRACT_REJECTED,
    )


@pytest.mark.parametrize("record_index", (0, 480, 959))
def test_first_middle_last_malformed_public_decoder_record_is_atomic(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
    monkeypatch: pytest.MonkeyPatch,
    record_index: int,
) -> None:
    decoded = mixed_scoring_fixture_v2.decoded_prediction_archive
    records = list(decoded.records)
    records[record_index] = _unchecked_copy(
        records[record_index],
        decision=object(),
    )
    malformed = _unchecked_copy(decoded, records=tuple(records))
    calls = 0

    def polluted_decoder(value: bytes) -> object:
        nonlocal calls
        calls += 1
        return malformed

    monkeypatch.setattr(
        mechanics_v2,
        "decode_public_recognizer_prediction_archive_v2",
        polluted_decoder,
    )
    result = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(mixed_scoring_fixture_v2)
    )
    assert calls == 1
    _assert_atomic_rejection(
        result,
        mechanics_v2.Unsealed960PredictionScoringReasonV2
        .PREDICTION_ARCHIVE_INVALID,
    )


def test_valid_other_archive_splice_is_atomic_cross_binding_rejection(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
    challenge_replaced_fixture_v2: _MixedScoringFixtureV2,
) -> None:
    result = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(
            mixed_scoring_fixture_v2,
            prediction_archive=(
                challenge_replaced_fixture_v2.prediction_archive_bytes
            ),
        )
    )
    _assert_atomic_rejection(
        result,
        mechanics_v2.Unsealed960PredictionScoringReasonV2
        .PREDICTION_ARCHIVE_BINDING_MISMATCH,
    )


def test_positive_only_nine_metrics_join_by_id_and_exclude_challenge(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
) -> None:
    fixture = mixed_scoring_fixture_v2
    result = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(fixture)
    )
    assert type(result) is mechanics_v2.Unsealed960PredictionScoringMechanicsV2
    assert result.disposition is (
        mechanics_v2.Unsealed960PredictionScoringDispositionV2
        .MECHANICS_COMPLETE_NOT_ACTUAL_EXECUTION
    )
    assert result.reason is (
        mechanics_v2.Unsealed960PredictionScoringReasonV2
        .CANONICAL_V2_MAIN_ROW_NINE_METRIC_MECHANICS_COMPLETE
    )
    assert result.version == (
        mechanics_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_VERSION
    )
    assert result.schema_id == (
        mechanics_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_SCHEMA_ID
    )
    assert result.policy_id == (
        mechanics_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_POLICY_ID
    )
    assert result.claim_level == (
        "NON_AUTHORITATIVE_SCORING_MECHANICS_ONLY"
    )
    assert (
        result.total_prediction_count,
        result.main_row_result_count,
        result.metric_eligible_main_row_count,
        result.control_row_without_frozen_metric_count,
        result.semantic_conflict_excluded_count,
    ) == (960, 720, 336, 384, 240)
    assert len(result.main_row_results) == 720
    assert len(result.metric_results) == 9
    assert tuple(item.input_row_id for item in result.main_row_results) == (
        fixture.partition_manifest.main_row_ids
    )
    assert tuple(item.input_row_id for item in result.main_row_results) == tuple(
        row.input_row_id for row in fixture.answer_manifest.main_answer_rows
    )
    assert set(item.input_row_id for item in result.main_row_results).isdisjoint(
        fixture.partition_manifest.semantic_conflict_row_ids
    )
    assert tuple(
        (
            item.metric_name,
            item.count_value
            if item.metric_name == "answerable_count"
            else item.success_count,
            item.observed_denominator,
        )
        for item in result.metric_results
    ) == fixture.expected_metrics
    assert tuple(item.metric_name for item in result.metric_results) == (
        EXPECTED_METRIC_NAMES
    )
    for item in result.metric_results:
        assert type(item.expected_denominator) is int
        assert type(item.observed_denominator) is int
        assert item.expected_denominator == item.observed_denominator
        assert not any(
            type(value) is float
            for value in (
                item.expected_denominator,
                item.observed_denominator,
                item.success_count,
                item.count_value,
            )
        )
        if item.metric_name == "answerable_count":
            assert item.success_count is None
            assert item.count_value == 240
        else:
            assert type(item.success_count) is int
            assert item.count_value is None
    assert all(getattr(result, name) is True for name in TRUE_RESULT_CLAIMS)
    assert all(getattr(result, name) is False for name in FALSE_RESULT_CLAIMS)
    assert result.gate_results == ()
    assert result.scale_regret_result is None
    assert result.bootstrap_result is None


def test_replacing_all_challenge_predictions_cannot_change_main_metrics(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
    challenge_replaced_fixture_v2: _MixedScoringFixtureV2,
) -> None:
    baseline = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(mixed_scoring_fixture_v2)
    )
    changed = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(challenge_replaced_fixture_v2)
    )
    assert type(baseline) is mechanics_v2.Unsealed960PredictionScoringMechanicsV2
    assert type(changed) is mechanics_v2.Unsealed960PredictionScoringMechanicsV2
    assert changed.prediction_archive_id != baseline.prediction_archive_id
    assert changed.partition_manifest_id != baseline.partition_manifest_id
    assert tuple(
        (
            item.metric_name,
            item.success_count,
            item.count_value,
            item.observed_denominator,
        )
        for item in changed.metric_results
    ) == tuple(
        (
            item.metric_name,
            item.success_count,
            item.count_value,
            item.observed_denominator,
        )
        for item in baseline.metric_results
    )
    assert tuple(
        (
            item.input_row_id,
            item.decision_exact,
            item.family_exact,
            item.binding_exact,
            item.scale_set_exact,
            item.joint_exact,
        )
        for item in changed.main_row_results
    ) == tuple(
        (
            item.input_row_id,
            item.decision_exact,
            item.family_exact,
            item.binding_exact,
            item.scale_set_exact,
            item.joint_exact,
        )
        for item in baseline.main_row_results
    )
    assert changed.challenge_scoring_performed is False
    assert changed.challenge_in_main_denominator is False


def test_main_rows_have_nine_explicit_outcomes_and_controls_have_no_metric(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
) -> None:
    result = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(mixed_scoring_fixture_v2)
    )
    assert type(result) is mechanics_v2.Unsealed960PredictionScoringMechanicsV2
    eligible = 0
    controls = 0
    for row in result.main_row_results:
        assert len(row.metric_outcomes) == 9
        assert tuple(item.metric_name for item in row.metric_outcomes) == (
            EXPECTED_METRIC_NAMES
        )
        for item in row.metric_outcomes:
            assert type(item.eligible) is bool
            if item.eligible:
                if item.metric_name == "answerable_count":
                    assert item.success is None
                else:
                    assert type(item.success) is bool
            else:
                assert item.success is None
        if row.case_type in (
            ANSWERABLE_CASE_TYPES
            | {Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE}
        ):
            eligible += 1
            assert row.metric_eligible is True
            assert any(item.eligible for item in row.metric_outcomes)
        else:
            controls += 1
            assert row.case_type in CONTROL_CASE_TYPES
            assert row.metric_eligible is False
            assert all(not item.eligible for item in row.metric_outcomes)
            assert all(item.success is None for item in row.metric_outcomes)
    assert (eligible, controls) == (336, 384)


def test_all_gate_inputs_results_regret_and_bootstrap_remain_missing(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
) -> None:
    result = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(mixed_scoring_fixture_v2)
    )
    assert type(result) is mechanics_v2.Unsealed960PredictionScoringMechanicsV2
    assert len(EXPECTED_OVERALL_GATE_NAMES) == 12
    assert len(EXPECTED_SLICE_GATE_REFERENCES) == 6
    assert result.gate_results == ()
    assert result.overall_gate_results_materialized is False
    assert result.formal_gate_evaluation_performed is False
    assert result.slice_gate_metrics_implemented is False
    assert result.scale_regret_evaluated is False
    assert result.scale_regret_result is None
    assert result.bootstrap_evaluated is False
    assert result.bootstrap_result is None
    assert {
        item.name
        for item in fields(mechanics_v2.Unsealed960PredictionScoringMechanicsV2)
    }.isdisjoint(
        {
            "overall_gate_inputs",
            "overall_gate_results",
            "slice_gate_inputs",
            "slice_gate_results",
            "scale_regret_input",
            "bootstrap_samples",
            "bootstrap_interval",
        }
    )


def test_public_decoder_and_formal_validator_are_each_called_exactly_once(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = mechanics_v2.decode_public_recognizer_prediction_archive_v2
    validator = mechanics_v2.validate_formal_unsealed_prediction_scoring_contract_v2
    counts = {"decoder": 0, "validator": 0}

    def counted_decoder(value: bytes) -> object:
        counts["decoder"] += 1
        return decoder(value)

    def counted_validator(**kwargs: object) -> object:
        counts["validator"] += 1
        return validator(**kwargs)

    monkeypatch.setattr(
        mechanics_v2,
        "decode_public_recognizer_prediction_archive_v2",
        counted_decoder,
    )
    monkeypatch.setattr(
        mechanics_v2,
        "validate_formal_unsealed_prediction_scoring_contract_v2",
        counted_validator,
    )
    result = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(mixed_scoring_fixture_v2)
    )
    assert type(result) is mechanics_v2.Unsealed960PredictionScoringMechanicsV2
    assert counts == {"decoder": 1, "validator": 1}


def test_result_is_fresh_and_caller_pollution_does_not_cross_calls(
    mixed_scoring_fixture_v2: _MixedScoringFixtureV2,
) -> None:
    first = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(mixed_scoring_fixture_v2)
    )
    assert type(first) is mechanics_v2.Unsealed960PredictionScoringMechanicsV2
    object.__setattr__(first.metric_results[0], "metric_name", "caller_pollution")
    object.__setattr__(first.main_row_results[0], "input_row_id", "caller_pollution")
    second = mechanics_v2.score_unsealed_960_prediction_scoring_mechanics_v2(
        **_score_kwargs(mixed_scoring_fixture_v2)
    )
    assert type(second) is mechanics_v2.Unsealed960PredictionScoringMechanicsV2
    assert second is not first
    assert second.metric_results[0] is not first.metric_results[0]
    assert second.metric_results[0].metric_name == "answerable_count"
    assert second.main_row_results[0] is not first.main_row_results[0]
    assert second.main_row_results[0].input_row_id == (
        mixed_scoring_fixture_v2.partition_manifest.main_row_ids[0]
    )
