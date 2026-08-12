"""Contract-only tests for formal unsealed V2 prediction scoring.

Fixtures in this file are synthetic and unbacked.  They may validate only the
precommitted scoring-contract mechanics; they are not an actual recognizer run,
formal scoring, capacity, effect, or C1 evidence.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields
import hashlib
import inspect
from pathlib import Path
import runpy

import pytest

import hegel_machine.phase2b_formal_unsealed_prediction_scoring_contract_v2 as scoring_v2
import hegel_machine.phase2b_strict_recognizer_cli_v2 as strict_v2
import hegel_machine.phase2b_unsealed_prediction_evaluator_v2 as evaluator_v2
from hegel_machine.phase2b_freeze_v1 import CanonicalFamilyId
from hegel_machine.hashing import canonical_json, stable_hash
from hegel_machine.phase2b_protocol import (
    Phase2BCaseType,
    salted_answer_commitment_sha256,
)
from hegel_machine.phase2b_recognizer_prediction_v2 import PredictionDecisionV2
from hegel_machine.phase2b_wire import RoleBinding


PUBLIC_FUNCTIONS = (
    "build_formal_unsealed_answer_manifest_v2",
    "frozen_formal_unsealed_prediction_scoring_contract_v2",
    "validate_formal_unsealed_prediction_scoring_contract_v2",
)
PUBLIC_TYPES = (
    "FormalUnsealedAnswerRowV2",
    "FormalUnsealedAnswerManifestV2",
    "FormalUnsealedMetricDefinitionV2",
    "FormalUnsealedPredictionScoringContractV2",
    "FormalUnsealedPredictionScoringContractValidationV2",
    "FormalUnsealedPredictionScoringContractRejectionV2",
)
PUBLIC_ENUMS = (
    "FormalUnsealedMetricKindV2",
    "FormalUnsealedPredictionScoringContractDispositionV2",
    "FormalUnsealedPredictionScoringContractReasonV2",
)
EXPECTED_PUBLIC_SURFACE = [
    "FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION",
    "FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL",
    "FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID",
    "FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID",
    *PUBLIC_ENUMS,
    *PUBLIC_TYPES,
    *PUBLIC_FUNCTIONS,
]
ANSWER_ROW_FIELDS = (
    "input_row_id",
    "case_type",
    "expected_decision",
    "canonical_family_id",
    "binding",
    "admissible_scale_ids",
    "answer_row_id",
)
ANSWER_MANIFEST_FIELDS = (
    "schema_version",
    "schema_id",
    "policy_id",
    "claim_level",
    "exact_freeze_id",
    "phase2b_protocol_id",
    "execution_freeze_manifest_id",
    "input_archive_id",
    "input_archive_sha256",
    "input_archive_version",
    "input_archive_policy_id",
    "batch_id",
    "batch_policy_id",
    "ordered_archive_input_row_ids_root",
    "main_row_ids_root",
    "semantic_conflict_row_ids_root",
    "partition_union_row_ids_root",
    "main_answer_rows",
    "main_answer_row_ids_root",
    "answer_manifest_sha256",
    "answer_manifest_id",
)
METRIC_DEFINITION_FIELDS = (
    "metric_name",
    "metric_kind",
    "denominator_case_types",
    "expected_denominator",
    "success_rule",
    "separately_reported",
    "metric_definition_id",
)
SCORING_CONTRACT_FIELDS = (
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "answer_row_schema_id",
    "answer_manifest_schema_id",
    "required_structural_receipt_type",
    "required_structural_evaluation_type",
    "required_partition_manifest_type",
    "main_row_count",
    "semantic_conflict_row_count",
    "case_type_counts",
    "metric_definitions",
    "set_valued_joint_rule",
    "commitment_opening_formula",
    "challenge_denominator_policy",
    "overall_gate_definitions",
    "slice_gate_definitions",
    "scale_regret_gate_definition",
    "bootstrap_reference",
    "bootstrap_evaluated",
    "overall_gate_metric_mapping",
    "wilson_method",
    "wilson_semantics",
    "wilson_confidence",
    "gate_inputs_implemented",
    "gate_results",
    "gates_executed",
    "contract_id",
)
TRUE_VALIDATION_CLAIMS = (
    "contract_identity_verified",
    "structural_receipt_binding_verified",
    "structural_evaluation_binding_verified",
    "partition_manifest_binding_verified",
    "evaluator_side_answer_schema_verified",
    "supplied_answer_commitment_opening_verified",
    "exact_main_answer_row_coverage_verified",
    "frozen_case_type_quota_verified",
    "nine_metric_definition_mechanics_verified",
    "challenge_excluded_from_main_denominator",
)
FALSE_VALIDATION_CLAIMS = (
    "challenge_in_main_denominator",
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
    "formal_gate_evaluation_performed",
    "metric_results_materialized",
    "scored_rows_materialized",
    "control_rejection_metrics_implemented",
    "slice_gate_metrics_implemented",
    "challenge_scoring_performed",
    "effect_evidence",
    "c1_exit_evidence",
)
VALIDATION_FIELDS = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "prediction_archive_id",
    "partition_manifest_id",
    "structural_receipt_id",
    "answer_manifest_id",
    "answer_manifest_sha256",
    "salted_answer_commitment_sha256",
    "main_row_count",
    "semantic_conflict_row_count",
    "answerable_row_count",
    "main_answer_row_ids_root",
    "ordered_archive_input_row_ids_root",
    *TRUE_VALIDATION_CLAIMS,
    *FALSE_VALIDATION_CLAIMS,
    "metric_definitions",
    "metric_results",
    "scored_rows",
)
REJECTION_FIELDS = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "validation",
    "metric_definitions",
    "metric_results",
    "scored_rows",
    "partial_output_published",
    *TRUE_VALIDATION_CLAIMS,
    *FALSE_VALIDATION_CLAIMS,
)
FORBIDDEN_PRECOMMIT_FIELDS = frozenset(
    {
        "prediction_archive_id",
        "prediction_archive_sha256",
        "prediction_archive_policy_id",
        "prediction_archive_version",
        "partition_manifest_id",
        "metric_results",
        "scored_rows",
    }
)
EXPECTED_METRIC_DENOMINATORS = (
    ("answerable_count", 240, False),
    ("family_exact_accuracy", 240, False),
    ("binding_exact_accuracy", 240, False),
    ("scale_set_accuracy", 240, False),
    ("unique_scale_accuracy", 228, False),
    ("joint_exact_accuracy", 240, False),
    ("abstention_specificity", 228, False),
    ("nonidentifiability_abstention_accuracy", 96, False),
    ("set_valued_answer_accuracy", 12, True),
)
EXPECTED_SUCCESS_RULES = (
    ("answerable_count", "eligible_case_count_equals_exact_denominator"),
    (
        "family_exact_accuracy",
        "positive_decision_and_canonical_family_exact",
    ),
    (
        "binding_exact_accuracy",
        "positive_decision_and_canonical_role_binding_tuple_exact",
    ),
    (
        "scale_set_accuracy",
        "positive_decision_and_canonical_admissible_scale_tuple_exact",
    ),
    (
        "unique_scale_accuracy",
        "unique_case_requires_ANSWER_and_exact_singleton_scale",
    ),
    (
        "joint_exact_accuracy",
        "expected_decision_family_binding_and_scale_all_exact_set_requires_ANSWER_SET",
    ),
    ("abstention_specificity", "unique_case_decision_is_not_ABSTAIN"),
    (
        "nonidentifiability_abstention_accuracy",
        "insufficient_or_nonidentifiable_requires_ABSTAIN_reason_not_scored",
    ),
    (
        "set_valued_answer_accuracy",
        "set_case_requires_ANSWER_SET_family_binding_and_full_scale_set_exact",
    ),
)
EXPECTED_METRIC_CASE_TYPES = (
    (
        "answerable_count",
        ("unique_scale_answerable", "admissible_scale_set_answerable"),
    ),
    (
        "family_exact_accuracy",
        ("unique_scale_answerable", "admissible_scale_set_answerable"),
    ),
    (
        "binding_exact_accuracy",
        ("unique_scale_answerable", "admissible_scale_set_answerable"),
    ),
    (
        "scale_set_accuracy",
        ("unique_scale_answerable", "admissible_scale_set_answerable"),
    ),
    ("unique_scale_accuracy", ("unique_scale_answerable",)),
    (
        "joint_exact_accuracy",
        ("unique_scale_answerable", "admissible_scale_set_answerable"),
    ),
    ("abstention_specificity", ("unique_scale_answerable",)),
    (
        "nonidentifiability_abstention_accuracy",
        ("insufficient_or_nonidentifiable",),
    ),
    (
        "set_valued_answer_accuracy",
        ("admissible_scale_set_answerable",),
    ),
)
EXPECTED_OVERALL_GATES = (
    ("family_exact", 0.90, 0.86),
    ("binding_exact", 0.90, 0.86),
    ("scale_set_accuracy", 0.87, 0.82),
    ("joint_exact", 0.85, 0.80),
    ("hard_negative_rejection", 0.95, 0.90),
    ("binding_counterfactual_rejection", 0.95, 0.90),
    ("scale_counterfactual_rejection", 0.93, 0.88),
    ("sign_or_invariant_break_rejection", 0.95, 0.90),
    ("abstention_specificity", 0.95, 0.90),
    ("fail_closed_rate", 1.0, None),
    ("preservation_consistency", 0.97, 0.94),
    ("nonidentifiable_scale_abstention", 0.95, 0.90),
)
EXPECTED_SLICE_GATES = (
    ("answerable_joint_exact", 0.80, 0.70, "family"),
    ("all_control_rejection", 0.88, 0.78, "family"),
    ("abstention_specificity", 0.85, 0.75, "family"),
    ("answerable_joint_exact", 0.80, 0.70, "scale"),
    ("all_control_rejection", 0.88, 0.78, "scale"),
    ("abstention_specificity", 0.85, 0.75, "scale"),
)
EXPECTED_SCALE_REGRET_GATE = (
    "normalized_scale_decision_regret",
    0.05,
    0.08,
)
EXPECTED_BOOTSTRAP_REFERENCE = (
    "paired_cluster_bootstrap",
    10_000,
    411_876_909_552_964_556,
    "sha256_domain_separated_uint64_be_first32_v1",
    2_611_585_425,
    "latent_base_case",
    "one_sided_95_percent_percentile",
)
EXPECTED_OVERALL_GATE_METRIC_MAPPING = (
    ("family_exact", "family_exact_accuracy"),
    ("binding_exact", "binding_exact_accuracy"),
    ("scale_set_accuracy", "scale_set_accuracy"),
    ("joint_exact", "joint_exact_accuracy"),
    ("hard_negative_rejection", None),
    ("binding_counterfactual_rejection", None),
    ("scale_counterfactual_rejection", None),
    ("sign_or_invariant_break_rejection", None),
    ("abstention_specificity", "abstention_specificity"),
    ("fail_closed_rate", None),
    ("preservation_consistency", None),
    (
        "nonidentifiable_scale_abstention",
        "nonidentifiability_abstention_accuracy",
    ),
)
EXPECTED_MAIN_CASE_COUNTS = {
    "unique_scale_answerable": 228,
    "admissible_scale_set_answerable": 12,
    "wrong_family_hard_negative": 96,
    "binding_counterfactual": 96,
    "scale_counterfactual": 96,
    "sign_or_invariant_break": 96,
    "insufficient_or_nonidentifiable": 96,
}
EXPECTED_WILSON_METHOD = "one_sided_wilson_lower_confidence_bound"
EXPECTED_WILSON_SEMANTICS = (
    "binary_success_count_over_exact_frozen_denominator_using_"
    "NormalDist_inv_cdf_confidence_no_gate_execution"
)
EXPECTED_POLICY_METRIC_SPECS = (
    (
        "answerable_count",
        "COUNT",
        ("unique_scale_answerable", "admissible_scale_set_answerable"),
        240,
        "eligible_case_count_equals_exact_denominator",
        False,
    ),
    (
        "family_exact_accuracy",
        "BINARY_ACCURACY",
        ("unique_scale_answerable", "admissible_scale_set_answerable"),
        240,
        "positive_decision_and_canonical_family_exact",
        False,
    ),
    (
        "binding_exact_accuracy",
        "BINARY_ACCURACY",
        ("unique_scale_answerable", "admissible_scale_set_answerable"),
        240,
        "positive_decision_and_canonical_role_binding_tuple_exact",
        False,
    ),
    (
        "scale_set_accuracy",
        "BINARY_ACCURACY",
        ("unique_scale_answerable", "admissible_scale_set_answerable"),
        240,
        "positive_decision_and_canonical_admissible_scale_tuple_exact",
        False,
    ),
    (
        "unique_scale_accuracy",
        "BINARY_ACCURACY",
        ("unique_scale_answerable",),
        228,
        "unique_case_requires_ANSWER_and_exact_singleton_scale",
        False,
    ),
    (
        "joint_exact_accuracy",
        "BINARY_ACCURACY",
        ("unique_scale_answerable", "admissible_scale_set_answerable"),
        240,
        "expected_decision_family_binding_and_scale_all_exact_set_requires_ANSWER_SET",
        False,
    ),
    (
        "abstention_specificity",
        "BINARY_ACCURACY",
        ("unique_scale_answerable",),
        228,
        "unique_case_decision_is_not_ABSTAIN",
        False,
    ),
    (
        "nonidentifiability_abstention_accuracy",
        "BINARY_ACCURACY",
        ("insufficient_or_nonidentifiable",),
        96,
        "insufficient_or_nonidentifiable_requires_ABSTAIN_reason_not_scored",
        False,
    ),
    (
        "set_valued_answer_accuracy",
        "BINARY_ACCURACY",
        ("admissible_scale_set_answerable",),
        12,
        "set_case_requires_ANSWER_SET_family_binding_and_full_scale_set_exact",
        True,
    ),
)
EXPECTED_ANSWER_MANIFEST_SCORING_MECHANICS = {
    "metric_specs": EXPECTED_POLICY_METRIC_SPECS,
    "set_valued_joint_rule": (
        "family_exact_and_binding_exact_and_scale_set_exact_and_ANSWER_SET"
    ),
    "overall_gates_referenced_not_executed": EXPECTED_OVERALL_GATES,
    "slice_gates_referenced_not_executed": EXPECTED_SLICE_GATES,
    "scale_regret_gate_referenced_not_executed": EXPECTED_SCALE_REGRET_GATE,
    "bootstrap_referenced_not_executed": EXPECTED_BOOTSTRAP_REFERENCE,
    "overall_gate_metric_mapping": EXPECTED_OVERALL_GATE_METRIC_MAPPING,
    "wilson_method": EXPECTED_WILSON_METHOD,
    "wilson_semantics": EXPECTED_WILSON_SEMANTICS,
    "wilson_confidence": 0.95,
    "gate_inputs_implemented": False,
    "gate_results": (),
    "gates_executed": False,
    "bootstrap_evaluated": False,
}

FORBIDDEN_DIRECT_IMPORT_MODULE_SUFFIXES = frozenset(
    {
        "phase2b_recognizer_prediction_archive_v1",
        "phase2b_recognizer_prediction_archive_v2",
        "phase2b_recognizer_input_archive_v1",
        "phase2b_recognizer_input_archive_v2",
        "phase2b_runner",
        "phase2b_evaluator",
        "subprocess",
        "socket",
        "requests",
    }
)
FORBIDDEN_DIRECT_CALL_NAMES = frozenset(
    {
        "build_recognizer_prediction_archive_v2",
        "decode_public_recognizer_input_archive_v2",
        "decode_public_recognizer_prediction_archive_v2",
        "evaluate_binary_gate",
        "evaluate_unsealed_prediction_archive_structure_v2",
        "one_sided_wilson_lower_bound",
        "paired_cluster_bootstrap",
        "recognize_public_input_row_v2",
        "run_recognizer",
        "score_prediction",
        "score_predictions",
        "verify_strict_recognizer_io_structure_v2",
    }
)


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

    called = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in (qualified_name(node.func),)
        if name is not None
    }
    return frozenset(imported), frozenset(called)


def _forbidden_direct_call(value: str) -> bool:
    folded = value.casefold()
    terminal = folded.rsplit(".", 1)[-1]
    module_parts = folded.split(".")[:-1]
    return (
        terminal in FORBIDDEN_DIRECT_CALL_NAMES
        or terminal.startswith(("decode_", "recognize_", "score_"))
        or terminal.endswith(("_decoder", "_scorer"))
        or "bootstrap" in terminal
        or "wilson_lower" in terminal
        or "gate" in terminal
        or any(
            part in FORBIDDEN_DIRECT_IMPORT_MODULE_SUFFIXES
            for part in module_parts
        )
    )


class _HostileFieldTouched(BaseException):
    """Sentinel proving an invalid field was used before exact-type closure."""


class _HostileText(str):
    def __eq__(self, other: object) -> bool:
        raise _HostileFieldTouched("hostile answer field reached equality")

    def __hash__(self) -> int:
        raise _HostileFieldTouched("hostile answer field reached hashing")

    def encode(self, *args: object, **kwargs: object) -> bytes:
        raise _HostileFieldTouched("hostile answer field reached encoding")

    def startswith(self, *args: object, **kwargs: object) -> bool:
        raise _HostileFieldTouched("hostile answer field reached prefix parsing")


def _hex_id(prefix: str, index: int) -> str:
    return f"{prefix}{index:064x}"


def _uuid4(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012x}"


def _synthetic_answer_row(
    *,
    index: int,
    case_type: Phase2BCaseType,
    input_row_id: str | None = None,
) -> scoring_v2.FormalUnsealedAnswerRowV2:
    if case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE:
        decision = PredictionDecisionV2.ANSWER
        family = CanonicalFamilyId.F01
        binding = (
            RoleBinding(
                role_id=_uuid4(10_000 + index),
                entity_id=_uuid4(20_000 + index),
            ),
        )
        scales = (_uuid4(30_000 + index),)
    elif case_type is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE:
        decision = PredictionDecisionV2.ANSWER_SET
        family = CanonicalFamilyId.F01
        binding = (
            RoleBinding(
                role_id=_uuid4(10_000 + index),
                entity_id=_uuid4(20_000 + index),
            ),
        )
        scales = tuple(
            sorted((_uuid4(30_000 + 2 * index), _uuid4(30_001 + 2 * index)))
        )
    else:
        decision = PredictionDecisionV2.ABSTAIN
        family = None
        binding = ()
        scales = ()
    if input_row_id is None:
        input_row_id = _hex_id(
            "phase2b_recognizer_input_row_v2_", index + 1
        )
    preimage = {
        "input_row_id": input_row_id,
        "case_type": case_type.value,
        "expected_decision": decision.value,
        "canonical_family_id": None if family is None else family.value,
        "binding": [
            {"role_id": item.role_id, "entity_id": item.entity_id}
            for item in binding
        ],
        "admissible_scale_ids": list(scales),
    }
    answer_row_id = (
        "phase2b_formal_unsealed_answer_row_v2_"
        + hashlib.sha256(
            b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_ROW/V2\x00"
            + canonical_json(preimage).encode("utf-8")
        ).hexdigest()
    )
    return scoring_v2.FormalUnsealedAnswerRowV2(
        input_row_id=input_row_id,
        case_type=case_type,
        expected_decision=decision,
        canonical_family_id=family,
        binding=binding,
        admissible_scale_ids=scales,
        answer_row_id=answer_row_id,
    )


def _unchecked_copy(value: object, **changes: object) -> object:
    copied = object.__new__(type(value))
    for item in fields(value):
        object.__setattr__(
            copied,
            item.name,
            changes.get(item.name, getattr(value, item.name)),
        )
    return copied


@dataclass(frozen=True, slots=True)
class _SyntheticContractFixtureV2:
    receipt: strict_v2.StrictRecognizerStructuralReceiptV2
    evaluation: evaluator_v2.UnsealedPredictionStructuralEvaluationV2
    partition: evaluator_v2.UnsealedPredictionPartitionManifestV2
    answer: scoring_v2.FormalUnsealedAnswerManifestV2
    salt: str
    commitment: str


def _answer_manifest_preimage(
    value: scoring_v2.FormalUnsealedAnswerManifestV2,
) -> dict[str, object]:
    row_mappings = [
        {
            "input_row_id": row.input_row_id,
            "case_type": row.case_type.value,
            "expected_decision": row.expected_decision.value,
            "canonical_family_id": (
                None
                if row.canonical_family_id is None
                else row.canonical_family_id.value
            ),
            "binding": [
                {"role_id": item.role_id, "entity_id": item.entity_id}
                for item in row.binding
            ],
            "admissible_scale_ids": list(row.admissible_scale_ids),
            "answer_row_id": row.answer_row_id,
        }
        for row in value.main_answer_rows
    ]
    return {
        name: (
            row_mappings
            if name == "main_answer_rows"
            else getattr(value, name)
        )
        for name in ANSWER_MANIFEST_FIELDS
        if name not in {"answer_manifest_sha256", "answer_manifest_id"}
    }


def _builder_kwargs(
    value: _SyntheticContractFixtureV2,
    **changes: object,
) -> dict[str, object]:
    answer = value.answer
    kwargs: dict[str, object] = {
        "input_archive_id": answer.input_archive_id,
        "input_archive_sha256": answer.input_archive_sha256,
        "input_archive_version": answer.input_archive_version,
        "input_archive_policy_id": answer.input_archive_policy_id,
        "batch_id": answer.batch_id,
        "batch_policy_id": answer.batch_policy_id,
        "exact_freeze_id": answer.exact_freeze_id,
        "phase2b_protocol_id": answer.phase2b_protocol_id,
        "execution_freeze_manifest_id": answer.execution_freeze_manifest_id,
        "ordered_archive_input_row_ids_root": (
            answer.ordered_archive_input_row_ids_root
        ),
        "main_row_ids_root": answer.main_row_ids_root,
        "semantic_conflict_row_ids_root": (
            answer.semantic_conflict_row_ids_root
        ),
        "partition_union_row_ids_root": answer.partition_union_row_ids_root,
        "main_answer_rows": answer.main_answer_rows,
    }
    kwargs.update(changes)
    return kwargs


def _validate_kwargs(
    value: _SyntheticContractFixtureV2,
    **changes: object,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "structural_receipt": value.receipt,
        "structural_evaluation": value.evaluation,
        "partition_manifest": value.partition,
        "answer_manifest": value.answer,
        "revealed_answer_manifest_sha256": value.answer.answer_manifest_sha256,
        "answer_commitment_salt": value.salt,
        "salted_answer_commitment_sha256": value.commitment,
    }
    kwargs.update(changes)
    return kwargs


def _assert_atomic_rejection(
    value: object,
    reason: scoring_v2.FormalUnsealedPredictionScoringContractReasonV2,
) -> None:
    assert type(value) is scoring_v2.FormalUnsealedPredictionScoringContractRejectionV2
    assert value.disposition is (
        scoring_v2.FormalUnsealedPredictionScoringContractDispositionV2.REJECTED
    )
    assert value.reason is reason
    assert value.validation is None
    assert value.metric_definitions == ()
    assert value.metric_results == ()
    assert value.scored_rows == ()
    assert value.partial_output_published is False
    for name in (*TRUE_VALIDATION_CLAIMS, *FALSE_VALIDATION_CLAIMS):
        assert getattr(value, name) is False


def _forbid_validator_finalization(
    monkeypatch: pytest.MonkeyPatch,
    *,
    message: str,
) -> None:
    def forbidden_finalize(*args: object, **kwargs: object) -> str:
        raise _HostileFieldTouched(message)

    monkeypatch.setattr(
        strict_v2.StrictRecognizerStructuralReceiptV2,
        "to_mapping",
        forbidden_finalize,
    )
    for name in (
        "_answer_row_id_v2",
        "_main_row_ids_root_v2",
        "_semantic_conflict_row_ids_root_v2",
        "_partition_union_row_ids_root_v2",
        "_partition_manifest_id_v2",
        "_answer_row_ids_root_v2",
        "_answer_manifest_sha_v2",
        "_salted_answer_commitment_sha256",
    ):
        monkeypatch.setattr(scoring_v2, name, forbidden_finalize)


@pytest.fixture(scope="module")
def synthetic_main_case_types() -> tuple[str, ...]:
    """Return the exact synthetic 720-row quota, without scoring any row."""

    rows = tuple(
        case_type
        for case_type, count in EXPECTED_MAIN_CASE_COUNTS.items()
        for _ in range(count)
    )
    assert len(rows) == 720
    return rows


@pytest.fixture(scope="module")
def synthetic_main_answer_rows() -> tuple[
    scoring_v2.FormalUnsealedAnswerRowV2, ...
]:
    rows: list[scoring_v2.FormalUnsealedAnswerRowV2] = []
    for case_type in Phase2BCaseType:
        count = EXPECTED_MAIN_CASE_COUNTS[case_type.value]
        rows.extend(
            _synthetic_answer_row(index=len(rows), case_type=case_type)
            for _ in range(count)
        )
    assert len(rows) == 720
    return tuple(rows)


@pytest.fixture(scope="module")
def synthetic_contract_v2() -> _SyntheticContractFixtureV2:
    """Build one unbacked contract fixture without running a recognizer/scorer."""

    namespace = runpy.run_path(
        str(
            Path(__file__).with_name(
                "test_phase2b_recognizer_prediction_archive_v2.py"
            )
        )
    )
    freeze = namespace["execution_freeze_manifest"].__wrapped__()
    archive_fixture = namespace["synthetic_archive"].__wrapped__(freeze)
    decoded = archive_fixture.decoded
    main_row_ids = tuple(sorted(decoded.input_row_ids[:720]))
    conflict_row_ids = tuple(sorted(decoded.input_row_ids[720:]))
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
        prediction_archive_sha256=hashlib.sha256(decoded.archive).hexdigest(),
        prediction_archive_version=decoded.schema_version,
        prediction_archive_policy_id=decoded.policy_id,
        batch_id=context.batch_id,
        batch_policy_id=context.batch_policy_id,
        run_context_id=context.context_id,
        execution_freeze_manifest_id=context.execution_freeze_manifest_id,
        protocol_id=context.protocol_id,
    )
    case_types = tuple(
        case_type
        for case_type in Phase2BCaseType
        for _ in range(EXPECTED_MAIN_CASE_COUNTS[case_type.value])
    )
    rows = tuple(
        _synthetic_answer_row(
            index=index,
            case_type=case_type,
            input_row_id=input_row_id,
        )
        for index, (input_row_id, case_type) in enumerate(
            zip(main_row_ids, case_types, strict=True)
        )
    )
    answer = scoring_v2.build_formal_unsealed_answer_manifest_v2(
        input_archive_id=receipt.input_archive_id,
        input_archive_sha256=receipt.input_archive_sha256,
        input_archive_version=receipt.input_archive_version,
        input_archive_policy_id=receipt.input_archive_policy_id,
        batch_id=receipt.batch_id,
        batch_policy_id=receipt.batch_policy_id,
        exact_freeze_id=partition.exact_freeze_id,
        phase2b_protocol_id=receipt.protocol_id,
        execution_freeze_manifest_id=receipt.execution_freeze_manifest_id,
        ordered_archive_input_row_ids_root=(
            partition.ordered_archive_input_row_ids_root
        ),
        main_row_ids_root=partition.main_row_ids_root,
        semantic_conflict_row_ids_root=(
            partition.semantic_conflict_row_ids_root
        ),
        partition_union_row_ids_root=partition.partition_union_row_ids_root,
        main_answer_rows=rows,
    )
    salt = "synthetic-contract-only-opening-salt-0123456789abcdef"
    commitment = salted_answer_commitment_sha256(
        answer.answer_manifest_sha256,
        salt,
    )
    return _SyntheticContractFixtureV2(
        receipt=receipt,
        evaluation=evaluation,
        partition=partition,
        answer=answer,
        salt=salt,
        commitment=commitment,
    )


def test_public_surface_names_are_exact_and_validator_only() -> None:
    for name in (*PUBLIC_FUNCTIONS, *PUBLIC_TYPES, *PUBLIC_ENUMS):
        assert hasattr(scoring_v2, name)
    validate = inspect.signature(
        scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2
    )
    assert "structural_receipt" in validate.parameters
    folded = " ".join(scoring_v2.__all__).casefold()
    for forbidden in ("score_predictions", "evaluate_predictions", "run_recognizer"):
        assert forbidden not in folded
    assert scoring_v2.FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION == (
        "hegel-machine-phase2b-formal-unsealed-prediction-scoring-contract/2"
    )
    assert scoring_v2.FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL == (
        "NON_AUTHORITATIVE_CONTRACT_ONLY"
    )
    assert scoring_v2.FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID == (
        "phase2b_formal_unsealed_prediction_scoring_contract_schema_v2_"
        "59c5ed77970bcbcd7e8dc00bed6407b876679fe274b8103bb0edc9941ab6503b"
    )
    assert scoring_v2.FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID == (
        "phase2b_formal_unsealed_prediction_scoring_contract_policy_v2_"
        "c80fc5ac1899fe2cf439cea8f4273a47e60d9e77b56bf5d4c0c303743ded96e1"
    )


def test_public_surface_signatures_and_field_manifests_are_exact() -> None:
    assert scoring_v2.__all__ == EXPECTED_PUBLIC_SURFACE
    build = inspect.signature(scoring_v2.build_formal_unsealed_answer_manifest_v2)
    freeze = inspect.signature(
        scoring_v2.frozen_formal_unsealed_prediction_scoring_contract_v2
    )
    validate = inspect.signature(
        scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2
    )
    assert tuple(build.parameters) == (
        "input_archive_id",
        "input_archive_sha256",
        "input_archive_version",
        "input_archive_policy_id",
        "batch_id",
        "batch_policy_id",
        "exact_freeze_id",
        "phase2b_protocol_id",
        "execution_freeze_manifest_id",
        "ordered_archive_input_row_ids_root",
        "main_row_ids_root",
        "semantic_conflict_row_ids_root",
        "partition_union_row_ids_root",
        "main_answer_rows",
    )
    assert tuple(freeze.parameters) == ()
    assert tuple(validate.parameters) == (
        "structural_receipt",
        "structural_evaluation",
        "partition_manifest",
        "answer_manifest",
        "revealed_answer_manifest_sha256",
        "answer_commitment_salt",
        "salted_answer_commitment_sha256",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for signature in (build, validate)
        for parameter in signature.parameters.values()
    )
    assert tuple(
        item.name for item in fields(scoring_v2.FormalUnsealedAnswerRowV2)
    ) == ANSWER_ROW_FIELDS
    assert tuple(
        item.name for item in fields(scoring_v2.FormalUnsealedAnswerManifestV2)
    ) == ANSWER_MANIFEST_FIELDS
    assert tuple(
        item.name for item in fields(scoring_v2.FormalUnsealedMetricDefinitionV2)
    ) == METRIC_DEFINITION_FIELDS
    assert tuple(
        item.name
        for item in fields(scoring_v2.FormalUnsealedPredictionScoringContractV2)
    ) == SCORING_CONTRACT_FIELDS
    assert tuple(
        item.name
        for item in fields(
            scoring_v2.FormalUnsealedPredictionScoringContractValidationV2
        )
    ) == VALIDATION_FIELDS
    assert tuple(
        item.name
        for item in fields(
            scoring_v2.FormalUnsealedPredictionScoringContractRejectionV2
        )
    ) == REJECTION_FIELDS


def test_answer_manifest_is_precommitted_and_has_no_postprediction_fields() -> None:
    manifest_fields = {
        item.name for item in fields(scoring_v2.FormalUnsealedAnswerManifestV2)
    }
    assert manifest_fields.isdisjoint(FORBIDDEN_PRECOMMIT_FIELDS)
    assert {
        "input_archive_id",
        "input_archive_sha256",
        "input_archive_version",
        "input_archive_policy_id",
        "batch_id",
        "batch_policy_id",
        "execution_freeze_manifest_id",
    } <= manifest_fields


def test_answer_manifest_policy_binds_exact_scoring_mechanics_literals(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    assert set(EXPECTED_ANSWER_MANIFEST_SCORING_MECHANICS) == {
        "metric_specs",
        "set_valued_joint_rule",
        "overall_gates_referenced_not_executed",
        "slice_gates_referenced_not_executed",
        "scale_regret_gate_referenced_not_executed",
        "bootstrap_referenced_not_executed",
        "overall_gate_metric_mapping",
        "wilson_method",
        "wilson_semantics",
        "wilson_confidence",
        "gate_inputs_implemented",
        "gate_results",
        "gates_executed",
        "bootstrap_evaluated",
    }
    policy_preimage = {
        "schema_id": (
            "phase2b_formal_unsealed_answer_manifest_schema_v2_"
            "3f427810029665a54854751b7d021a77c4d5f874b7df1992d50434b7108d32f0"
        ),
        "claim_level": "NON_AUTHORITATIVE_CONTRACT_ONLY",
        "counts": (720, 240, 960),
        "case_type_counts": (
            ("unique_scale_answerable", 228),
            ("admissible_scale_set_answerable", 12),
            ("wrong_family_hard_negative", 96),
            ("binding_counterfactual", 96),
            ("scale_counterfactual", 96),
            ("sign_or_invariant_break", 96),
            ("insufficient_or_nonidentifiable", 96),
        ),
        "roots": {
            "answer_rows_domain": (
                b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_ROW_IDS/V2\x00".hex()
            ),
            "main_domain": b"HEGEL/PHASE2B/UNSEALED/MAIN_ROWS/V2\x00".hex(),
            "semantic_conflict_domain": (
                b"HEGEL/PHASE2B/UNSEALED/SEMANTIC_CONFLICT_ROWS/V2\x00".hex()
            ),
            "union_domain": (
                b"HEGEL/PHASE2B/UNSEALED/PARTITION_UNION_ROWS/V2\x00".hex()
            ),
            "ordered_domain": (
                b"HEGEL/PHASE2B/PREDICTION_INPUT_ROWS/V2\x00".hex()
            ),
        },
        "precommit": (
            "bind_upstream_input_batch_protocol_exact_and_execution_freezes",
            "bind_partition_roots_without_prediction_or_partition_ids",
            "no_prediction_metric_or_scored_row_fields",
        ),
        "scoring_mechanics": EXPECTED_ANSWER_MANIFEST_SCORING_MECHANICS,
    }
    expected_policy_id = stable_hash(
        policy_preimage,
        prefix="phase2b_formal_unsealed_answer_manifest_policy_v2_",
    )
    assert expected_policy_id == (
        "phase2b_formal_unsealed_answer_manifest_policy_v2_"
        "be684716aadb4bb6cced67348233d0c6ca78d7e0c98c6df2542bcc1787c50f1e"
    )
    assert synthetic_contract_v2.answer.policy_id == expected_policy_id


def test_synthetic_main_quota_recomputes_all_nine_denominators(
    synthetic_main_case_types: tuple[str, ...],
) -> None:
    counts = {
        case_type: synthetic_main_case_types.count(case_type)
        for case_type in EXPECTED_MAIN_CASE_COUNTS
    }
    assert counts == EXPECTED_MAIN_CASE_COUNTS
    answerable = counts["unique_scale_answerable"] + counts[
        "admissible_scale_set_answerable"
    ]
    assert answerable == 240
    assert counts["unique_scale_answerable"] == 228
    assert counts["insufficient_or_nonidentifiable"] == 96
    assert counts["admissible_scale_set_answerable"] == 12


def test_synthetic_answer_rows_lock_exact_case_semantics(
    synthetic_main_answer_rows: tuple[
        scoring_v2.FormalUnsealedAnswerRowV2, ...
    ],
) -> None:
    assert {
        case_type: sum(row.case_type is case_type for row in synthetic_main_answer_rows)
        for case_type in Phase2BCaseType
    } == {
        case_type: EXPECTED_MAIN_CASE_COUNTS[case_type.value]
        for case_type in Phase2BCaseType
    }
    for row in synthetic_main_answer_rows:
        assert type(row) is scoring_v2.FormalUnsealedAnswerRowV2
        assert type(row.case_type) is Phase2BCaseType
        assert type(row.expected_decision) is PredictionDecisionV2
        if row.case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE:
            assert row.expected_decision is PredictionDecisionV2.ANSWER
            assert type(row.canonical_family_id) is CanonicalFamilyId
            assert row.binding and all(type(item) is RoleBinding for item in row.binding)
            assert len(row.admissible_scale_ids) == 1
        elif row.case_type is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE:
            assert row.expected_decision is PredictionDecisionV2.ANSWER_SET
            assert type(row.canonical_family_id) is CanonicalFamilyId
            assert row.binding and all(type(item) is RoleBinding for item in row.binding)
            assert len(row.admissible_scale_ids) >= 2
            assert row.admissible_scale_ids == tuple(
                sorted(set(row.admissible_scale_ids))
            )
        else:
            assert row.expected_decision is PredictionDecisionV2.ABSTAIN
            assert row.canonical_family_id is None
            assert row.binding == ()
            assert row.admissible_scale_ids == ()


def test_public_answer_row_construction_does_not_make_pollution_valid(
    synthetic_main_answer_rows: tuple[
        scoring_v2.FormalUnsealedAnswerRowV2, ...
    ],
) -> None:
    clean = synthetic_main_answer_rows[0]
    polluted = scoring_v2.FormalUnsealedAnswerRowV2(
        input_row_id=_HostileText(clean.input_row_id),
        case_type=clean.case_type,
        expected_decision=clean.expected_decision,
        canonical_family_id=clean.canonical_family_id,
        binding=clean.binding,
        admissible_scale_ids=clean.admissible_scale_ids,
        answer_row_id=clean.answer_row_id,
    )
    # The public dataclass is a transport object, not a validation capability.
    assert type(polluted.input_row_id) is _HostileText


def test_frozen_contract_locks_nine_metric_definitions_and_executes_no_gate() -> None:
    contract = scoring_v2.frozen_formal_unsealed_prediction_scoring_contract_v2()
    observed_denominators = tuple(
        (
            definition.metric_name,
            definition.expected_denominator,
            definition.separately_reported,
        )
        for definition in contract.metric_definitions
    )
    observed_case_types = tuple(
        (
            definition.metric_name,
            tuple(case_type.value for case_type in definition.denominator_case_types),
        )
        for definition in contract.metric_definitions
    )
    observed_success_rules = tuple(
        (definition.metric_name, definition.success_rule)
        for definition in contract.metric_definitions
    )
    assert observed_denominators == EXPECTED_METRIC_DENOMINATORS
    assert observed_case_types == EXPECTED_METRIC_CASE_TYPES
    assert observed_success_rules == EXPECTED_SUCCESS_RULES
    assert tuple(definition.metric_kind.value for definition in contract.metric_definitions) == (
        "COUNT",
        *("BINARY_ACCURACY",) * 8,
    )
    assert contract.set_valued_joint_rule == (
        "family_exact_and_binding_exact_and_scale_set_exact_and_ANSWER_SET"
    )
    assert contract.main_row_count == 720
    assert contract.semantic_conflict_row_count == 240
    assert contract.challenge_denominator_policy == (
        "semantic_conflict_240_excluded_from_all_main_metric_denominators_and_threshold_tuning"
    )
    assert contract.gates_executed is False
    assert contract.contract_id == (
        "phase2b_formal_unsealed_prediction_scoring_contract_v2_"
        "37fce52fac6287a16d1925e76424d4d5b4e05fdcc552bc093d75d60a601d183e"
    )


def test_thresholds_wilson_and_scale_regret_are_referenced_not_executed() -> None:
    contract = scoring_v2.frozen_formal_unsealed_prediction_scoring_contract_v2()
    assert contract.overall_gate_definitions == EXPECTED_OVERALL_GATES
    assert contract.slice_gate_definitions == EXPECTED_SLICE_GATES
    assert contract.scale_regret_gate_definition == EXPECTED_SCALE_REGRET_GATE
    assert contract.bootstrap_reference == EXPECTED_BOOTSTRAP_REFERENCE
    assert contract.bootstrap_evaluated is False
    assert contract.overall_gate_metric_mapping == EXPECTED_OVERALL_GATE_METRIC_MAPPING
    assert type(contract.wilson_confidence) is float
    assert contract.wilson_confidence == 0.95
    assert contract.wilson_method == EXPECTED_WILSON_METHOD
    assert contract.wilson_semantics == EXPECTED_WILSON_SEMANTICS
    assert contract.gate_inputs_implemented is False
    assert type(contract.gate_results) is tuple and contract.gate_results == ()
    assert contract.gates_executed is False
    assert {
        item.name
        for item in fields(scoring_v2.FormalUnsealedPredictionScoringContractV2)
    }.isdisjoint(
        {
            "scale_cell_labels",
            "margin_stratum_labels",
            "preservation_pair_labels",
            "scale_slice_results",
            "scale_regret_result",
        }
    )


def test_contract_source_has_no_decoder_scorer_runtime_or_gate_execution() -> None:
    source_path = Path(scoring_v2.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    imported, called = _ast_direct_imports_and_calls(source)
    assert not any(
        target.rsplit(".", 1)[-1]
        in FORBIDDEN_DIRECT_IMPORT_MODULE_SUFFIXES
        for target in imported
    )
    assert not any(
        target.rsplit(".", 1)[-1] in FORBIDDEN_DIRECT_CALL_NAMES
        for target in imported
    )
    assert not any(_forbidden_direct_call(target) for target in called)


def test_ast_guard_resolves_import_aliases_and_full_attribute_call_chains() -> None:
    synthetic = """
import pkg.phase2b_runner as runner_alias
from pkg.archive import decode_public_recognizer_prediction_archive_v2 as decode_alias
from pkg.gates import evaluate_binary_gate as gate_alias
runner_alias.deep.run_recognizer()
decode_alias()
gate_alias()
object_alias.deep.scorer.score_predictions()
"""
    imported, called = _ast_direct_imports_and_calls(synthetic)
    assert "pkg.phase2b_runner" in imported
    assert (
        "pkg.archive.decode_public_recognizer_prediction_archive_v2"
        in imported
    )
    assert "pkg.phase2b_runner.deep.run_recognizer" in called
    assert "pkg.archive.decode_public_recognizer_prediction_archive_v2" in called
    assert "pkg.gates.evaluate_binary_gate" in called
    assert "object_alias.deep.scorer.score_predictions" in called
    assert all(_forbidden_direct_call(target) for target in called)


def test_answer_builder_independently_replays_nonself_manifest_commitment(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    answer = synthetic_contract_v2.answer
    partition = synthetic_contract_v2.partition
    assert type(answer) is scoring_v2.FormalUnsealedAnswerManifestV2
    assert tuple(row.input_row_id for row in answer.main_answer_rows) == (
        partition.main_row_ids
    )
    assert answer.execution_freeze_manifest_id == (
        synthetic_contract_v2.receipt.execution_freeze_manifest_id
    )
    assert answer.main_row_ids_root == partition.main_row_ids_root
    assert answer.semantic_conflict_row_ids_root == (
        partition.semantic_conflict_row_ids_root
    )
    assert answer.partition_union_row_ids_root == partition.partition_union_row_ids_root
    assert answer.ordered_archive_input_row_ids_root == (
        partition.ordered_archive_input_row_ids_root
    )

    answer_ids = tuple(row.answer_row_id for row in answer.main_answer_rows)
    digest = hashlib.sha256()
    digest.update(b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_ROW_IDS/V2\x00")
    digest.update((720).to_bytes(4, "big"))
    for answer_id in answer_ids:
        encoded = answer_id.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    assert answer.main_answer_row_ids_root == (
        "phase2b_formal_unsealed_answer_rows_v2_" + digest.hexdigest()
    )

    preimage = _answer_manifest_preimage(answer)
    assert set(preimage) == set(ANSWER_MANIFEST_FIELDS) - {
        "answer_manifest_sha256",
        "answer_manifest_id",
    }
    expected_sha = hashlib.sha256(
        b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_MANIFEST/V2\x00"
        + canonical_json(preimage).encode("utf-8")
    ).hexdigest()
    assert answer.answer_manifest_sha256 == expected_sha
    assert answer.answer_manifest_id == (
        "phase2b_formal_unsealed_answer_manifest_v2_" + expected_sha
    )


def test_builder_privately_addresses_empty_public_rows_without_mutating_inputs(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    public_rows = tuple(
        scoring_v2.FormalUnsealedAnswerRowV2(
            input_row_id=row.input_row_id,
            case_type=row.case_type,
            expected_decision=row.expected_decision,
            canonical_family_id=row.canonical_family_id,
            binding=row.binding,
            admissible_scale_ids=row.admissible_scale_ids,
        )
        for row in synthetic_contract_v2.answer.main_answer_rows
    )
    assert all(row.answer_row_id == "" for row in public_rows)
    rebuilt = scoring_v2.build_formal_unsealed_answer_manifest_v2(
        **_builder_kwargs(
            synthetic_contract_v2,
            main_answer_rows=public_rows,
        )
    )
    assert all(row.answer_row_id == "" for row in public_rows)
    assert all(
        issued.answer_row_id.startswith(
            "phase2b_formal_unsealed_answer_row_v2_"
        )
        and issued is not supplied
        for supplied, issued in zip(
            public_rows,
            rebuilt.main_answer_rows,
            strict=True,
        )
    )
    assert rebuilt.main_answer_rows == synthetic_contract_v2.answer.main_answer_rows
    assert rebuilt.answer_manifest_sha256 == (
        synthetic_contract_v2.answer.answer_manifest_sha256
    )


def test_polluted_getter_metric_cannot_contaminate_next_getter_or_validation(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    first = scoring_v2.frozen_formal_unsealed_prediction_scoring_contract_v2()
    polluted_metric = first.metric_definitions[0]
    object.__setattr__(polluted_metric, "metric_name", "caller_pollution")
    second = scoring_v2.frozen_formal_unsealed_prediction_scoring_contract_v2()
    assert second is not first
    assert second.metric_definitions[0] is not polluted_metric
    assert second.metric_definitions[0].metric_name == "answerable_count"

    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(synthetic_contract_v2)
    )
    assert type(result) is scoring_v2.FormalUnsealedPredictionScoringContractValidationV2
    assert result.metric_definitions[0] is not polluted_metric
    assert result.metric_definitions[0] is not second.metric_definitions[0]
    assert result.metric_definitions[0].metric_name == "answerable_count"


def test_contract_private_result_types_cannot_be_publicly_constructed() -> None:
    for value_type in (
        scoring_v2.FormalUnsealedAnswerManifestV2,
        scoring_v2.FormalUnsealedMetricDefinitionV2,
        scoring_v2.FormalUnsealedPredictionScoringContractV2,
        scoring_v2.FormalUnsealedPredictionScoringContractValidationV2,
        scoring_v2.FormalUnsealedPredictionScoringContractRejectionV2,
    ):
        with pytest.raises(TypeError, match="privately issued"):
            value_type()


def test_validator_positive_is_contract_only_and_all_evidence_claims_stay_false(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(synthetic_contract_v2)
    )
    assert type(result) is scoring_v2.FormalUnsealedPredictionScoringContractValidationV2
    assert result.disposition is (
        scoring_v2.FormalUnsealedPredictionScoringContractDispositionV2
        .CONTRACT_BINDING_COMPLETE_NOT_SCORED
    )
    assert result.reason is (
        scoring_v2.FormalUnsealedPredictionScoringContractReasonV2
        .CONTRACT_BINDING_VERIFIED
    )
    assert result.prediction_archive_id == synthetic_contract_v2.evaluation.prediction_archive_id
    assert result.partition_manifest_id == synthetic_contract_v2.partition.manifest_id
    assert result.structural_receipt_id == synthetic_contract_v2.receipt.receipt_id
    assert result.answer_manifest_id == synthetic_contract_v2.answer.answer_manifest_id
    assert result.answer_manifest_sha256 == (
        synthetic_contract_v2.answer.answer_manifest_sha256
    )
    assert result.salted_answer_commitment_sha256 == synthetic_contract_v2.commitment
    assert (result.main_row_count, result.semantic_conflict_row_count) == (720, 240)
    assert result.answerable_row_count == 240
    assert all(getattr(result, name) is True for name in TRUE_VALIDATION_CLAIMS)
    assert all(getattr(result, name) is False for name in FALSE_VALIDATION_CLAIMS)
    assert result.metric_definitions == (
        scoring_v2.frozen_formal_unsealed_prediction_scoring_contract_v2()
        .metric_definitions
    )
    assert result.metric_results == ()
    assert result.scored_rows == ()


@pytest.mark.parametrize(
    "field_name",
    (
        "structural_receipt",
        "structural_evaluation",
        "partition_manifest",
        "answer_manifest",
    ),
)
def test_validator_wrong_top_level_type_is_atomic(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
    field_name: str,
) -> None:
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(synthetic_contract_v2, **{field_name: object()})
    )
    _assert_atomic_rejection(
        result,
        scoring_v2.FormalUnsealedPredictionScoringContractReasonV2.WRONG_INPUT_TYPE,
    )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        (
            "input_archive_id",
            _hex_id("phase2b_recognizer_input_archive_v2_", 900_001),
        ),
        ("input_archive_sha256", f"{900_002:064x}"),
        ("batch_id", _hex_id("phase2b_trusted_wire_batch_v2_", 900_003)),
        (
            "execution_freeze_manifest_id",
            _hex_id("phase2b_execution_freeze_", 900_004),
        ),
        (
            "ordered_archive_input_row_ids_root",
            _hex_id("phase2b_prediction_input_rows_v2_", 900_006),
        ),
        (
            "semantic_conflict_row_ids_root",
            _hex_id("phase2b_unsealed_semantic_conflict_rows_v2_", 900_007),
        ),
        (
            "partition_union_row_ids_root",
            _hex_id("phase2b_unsealed_partition_union_rows_v2_", 900_008),
        ),
    ),
)
def test_locally_valid_answer_manifest_splices_fail_cross_object_binding(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
    field_name: str,
    replacement: str,
) -> None:
    spliced = scoring_v2.build_formal_unsealed_answer_manifest_v2(
        **_builder_kwargs(
            synthetic_contract_v2,
            **{field_name: replacement},
        )
    )
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(synthetic_contract_v2, answer_manifest=spliced)
    )
    _assert_atomic_rejection(
        result,
        scoring_v2.FormalUnsealedPredictionScoringContractReasonV2.IDENTITY_MISMATCH,
    )


@pytest.mark.parametrize(
    ("field_name", "prefix"),
    (
        ("prediction_archive_id", "phase2b_recognizer_prediction_archive_v2_"),
        ("partition_manifest_id", "phase2b_unsealed_prediction_partition_v2_"),
        ("main_row_ids_root", "phase2b_unsealed_main_rows_v2_"),
        (
            "semantic_conflict_row_ids_root",
            "phase2b_unsealed_semantic_conflict_rows_v2_",
        ),
        ("partition_union_row_ids_root", "phase2b_unsealed_partition_union_rows_v2_"),
        ("ordered_archive_input_row_ids_root", "phase2b_prediction_input_rows_v2_"),
    ),
)
def test_structural_evaluation_splices_fail_cross_object_binding(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
    field_name: str,
    prefix: str,
) -> None:
    spliced = _unchecked_copy(
        synthetic_contract_v2.evaluation,
        **{field_name: _hex_id(prefix, 910_000)},
    )
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(synthetic_contract_v2, structural_evaluation=spliced)
    )
    _assert_atomic_rejection(
        result,
        scoring_v2.FormalUnsealedPredictionScoringContractReasonV2.IDENTITY_MISMATCH,
    )


@pytest.mark.parametrize(
    ("object_name", "field_name", "replacement", "expected_reason"),
    (
        ("receipt", "structural_input_archive_verified", False, "CROSS_VERSION_INPUT"),
        ("receipt", "runtime_executed", True, "CROSS_VERSION_INPUT"),
        (
            "evaluation",
            "structural_completeness_verified",
            False,
            "STRUCTURAL_EVALUATION_NOT_COMPLETE",
        ),
        (
            "evaluation",
            "prediction_scored",
            True,
            "STRUCTURAL_EVALUATION_NOT_COMPLETE",
        ),
    ),
)
def test_upstream_claim_pollution_is_atomic_and_cannot_authorize_scoring(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
    object_name: str,
    field_name: str,
    replacement: bool,
    expected_reason: str,
) -> None:
    polluted = _unchecked_copy(
        getattr(synthetic_contract_v2, object_name),
        **{field_name: replacement},
    )
    argument = (
        "structural_receipt" if object_name == "receipt" else "structural_evaluation"
    )
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(synthetic_contract_v2, **{argument: polluted})
    )
    _assert_atomic_rejection(
        result,
        getattr(
            scoring_v2.FormalUnsealedPredictionScoringContractReasonV2,
            expected_reason,
        ),
    )


@pytest.mark.parametrize(
    ("change", "expected_message"),
    (
        ({"main_answer_rows": ()}, "exact 720"),
        ({"main_answer_rows": []}, "exact 720"),
        (
            {
                "main_row_ids_root": _hex_id(
                    "phase2b_unsealed_main_rows_v2_", 920_001
                )
            },
            "main partition root",
        ),
    ),
)
def test_answer_builder_rejects_order_count_and_root_drift(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
    change: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=expected_message):
        scoring_v2.build_formal_unsealed_answer_manifest_v2(
            **_builder_kwargs(synthetic_contract_v2, **change)
        )


def test_answer_builder_rejects_frozen_protocol_splice_locally(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    with pytest.raises(ValueError, match="frozen identity"):
        scoring_v2.build_formal_unsealed_answer_manifest_v2(
            **_builder_kwargs(
                synthetic_contract_v2,
                phase2b_protocol_id=_hex_id("phase2b_protocol_", 900_005),
            )
        )


def test_answer_builder_rejects_unsorted_rows_and_case_semantic_drift(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    rows = synthetic_contract_v2.answer.main_answer_rows
    with pytest.raises(ValueError, match="not sorted unique"):
        scoring_v2.build_formal_unsealed_answer_manifest_v2(
            **_builder_kwargs(
                synthetic_contract_v2,
                main_answer_rows=tuple(reversed(rows)),
            )
        )
    for case_type in Phase2BCaseType:
        index = next(
            position
            for position, row in enumerate(rows)
            if row.case_type is case_type
        )
        row = rows[index]
        if case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE:
            polluted = _unchecked_copy(row, expected_decision=PredictionDecisionV2.ANSWER_SET)
        elif case_type is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE:
            polluted = _unchecked_copy(row, expected_decision=PredictionDecisionV2.ANSWER)
        else:
            polluted = _unchecked_copy(row, canonical_family_id=CanonicalFamilyId.F01)
        changed_rows = (*rows[:index], polluted, *rows[index + 1 :])
        with pytest.raises(ValueError, match="case/decision payload"):
            scoring_v2.build_formal_unsealed_answer_manifest_v2(
                **_builder_kwargs(
                    synthetic_contract_v2,
                    main_answer_rows=changed_rows,
                )
            )


def test_validator_reports_quota_drift_as_atomic_case_type_rejection(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    rows = synthetic_contract_v2.answer.main_answer_rows
    first = rows[0]
    changed = _synthetic_answer_row(
        index=930_001,
        case_type=Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE,
        input_row_id=first.input_row_id,
    )
    malformed = _unchecked_copy(
        synthetic_contract_v2.answer,
        main_answer_rows=(changed, *rows[1:]),
    )
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(synthetic_contract_v2, answer_manifest=malformed)
    )
    _assert_atomic_rejection(
        result,
        scoring_v2.FormalUnsealedPredictionScoringContractReasonV2
        .CASE_TYPE_QUOTA_MISMATCH,
    )


@pytest.mark.parametrize(
    "opening_change",
    (
        {"revealed_answer_manifest_sha256": f"{940_001:064x}"},
        {"salted_answer_commitment_sha256": f"{940_002:064x}"},
        {"answer_commitment_salt": "too-short"},
        {"answer_commitment_salt": _HostileText("x" * 40)},
    ),
)
def test_opening_mismatch_is_atomic_and_never_claims_timing_or_authority(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
    opening_change: dict[str, object],
) -> None:
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(synthetic_contract_v2, **opening_change)
    )
    _assert_atomic_rejection(
        result,
        scoring_v2.FormalUnsealedPredictionScoringContractReasonV2
        .ANSWER_COMMITMENT_OPENING_INVALID,
    )


def test_revealed_sha_must_equal_stored_before_commitment_function_is_called(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    def forbidden_commitment(*args: object, **kwargs: object) -> str:
        raise _HostileFieldTouched("commitment computed before reveal parity")

    monkeypatch.setattr(
        scoring_v2,
        "_salted_answer_commitment_sha256",
        forbidden_commitment,
    )
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(
            synthetic_contract_v2,
            revealed_answer_manifest_sha256=f"{950_001:064x}",
        )
    )
    _assert_atomic_rejection(
        result,
        scoring_v2.FormalUnsealedPredictionScoringContractReasonV2
        .ANSWER_COMMITMENT_OPENING_INVALID,
    )


def test_hostile_answer_fields_close_before_row_or_manifest_hash(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    rows = synthetic_contract_v2.answer.main_answer_rows
    hostile_row = _unchecked_copy(rows[0], input_row_id=_HostileText(rows[0].input_row_id))

    def forbidden_hash(*args: object, **kwargs: object) -> str:
        raise _HostileFieldTouched("invalid row reached a content hash")

    monkeypatch.setattr(scoring_v2, "_answer_row_id_v2", forbidden_hash)
    monkeypatch.setattr(scoring_v2, "_answer_manifest_sha_v2", forbidden_hash)
    with pytest.raises(TypeError, match="exact text"):
        scoring_v2.build_formal_unsealed_answer_manifest_v2(
            **_builder_kwargs(
                synthetic_contract_v2,
                main_answer_rows=(hostile_row, *rows[1:]),
            )
        )


def test_hostile_upstream_fields_close_to_atomic_rejection_before_compare_or_hash(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    hostile_receipt = _unchecked_copy(
        synthetic_contract_v2.receipt,
        input_archive_id=_HostileText(
            synthetic_contract_v2.receipt.input_archive_id
        ),
    )
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(synthetic_contract_v2, structural_receipt=hostile_receipt)
    )
    _assert_atomic_rejection(
        result,
        scoring_v2.FormalUnsealedPredictionScoringContractReasonV2.CROSS_VERSION_INPUT,
    )

    hostile_partition = _unchecked_copy(
        synthetic_contract_v2.partition,
        main_row_ids=type(
            "HostileTuple",
            (tuple,),
            {"__iter__": lambda self: (_ for _ in ()).throw(_HostileFieldTouched())},
        )(synthetic_contract_v2.partition.main_row_ids),
    )
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(synthetic_contract_v2, partition_manifest=hostile_partition)
    )
    _assert_atomic_rejection(
        result,
        scoring_v2.FormalUnsealedPredictionScoringContractReasonV2
        .PARTITION_MANIFEST_INVALID,
    )


def test_malformed_nonempty_answer_id_rejects_before_content_hash(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    rows = synthetic_contract_v2.answer.main_answer_rows
    malformed = _unchecked_copy(rows[0], answer_row_id="not-a-prefixed-digest")

    def forbidden_content_hash(*args: object, **kwargs: object) -> str:
        raise _HostileFieldTouched("malformed answer ID reached content hash")

    monkeypatch.setattr(scoring_v2, "_answer_row_id_v2", forbidden_content_hash)
    with pytest.raises(ValueError, match="prefix drift"):
        scoring_v2.build_formal_unsealed_answer_manifest_v2(
            **_builder_kwargs(
                synthetic_contract_v2,
                main_answer_rows=(malformed, *rows[1:]),
            )
        )


def test_manifest_fresh_copies_rows_and_bindings_before_caller_pollution(
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    supplied_rows = tuple(
        scoring_v2.FormalUnsealedAnswerRowV2(
            input_row_id=row.input_row_id,
            case_type=row.case_type,
            expected_decision=row.expected_decision,
            canonical_family_id=row.canonical_family_id,
            binding=tuple(
                RoleBinding(
                    role_id=item.role_id,
                    entity_id=item.entity_id,
                )
                for item in row.binding
            ),
            admissible_scale_ids=tuple(row.admissible_scale_ids),
            answer_row_id=row.answer_row_id,
        )
        for row in synthetic_contract_v2.answer.main_answer_rows
    )
    manifest = scoring_v2.build_formal_unsealed_answer_manifest_v2(
        **_builder_kwargs(
            synthetic_contract_v2,
            main_answer_rows=supplied_rows,
        )
    )
    assert all(
        stored is not supplied
        for supplied, stored in zip(
            supplied_rows,
            manifest.main_answer_rows,
            strict=True,
        )
    )
    positive_index = next(
        index for index, row in enumerate(supplied_rows) if row.binding
    )
    supplied = supplied_rows[positive_index]
    stored = manifest.main_answer_rows[positive_index]
    assert stored.binding is not supplied.binding
    assert all(
        stored_item is not supplied_item
        for supplied_item, stored_item in zip(
            supplied.binding,
            stored.binding,
            strict=True,
        )
    )
    original_input_id = stored.input_row_id
    original_entity_id = stored.binding[0].entity_id
    original_preimage = _answer_manifest_preimage(manifest)
    original_sha = manifest.answer_manifest_sha256

    object.__setattr__(
        supplied,
        "input_row_id",
        _hex_id("phase2b_recognizer_input_row_v2_", 990_001),
    )
    object.__setattr__(supplied.binding[0], "entity_id", _uuid4(990_002))
    assert stored.input_row_id == original_input_id
    assert stored.binding[0].entity_id == original_entity_id
    assert _answer_manifest_preimage(manifest) == original_preimage
    assert manifest.answer_manifest_sha256 == original_sha

    salt = "post-build-caller-pollution-isolation-salt-0123456789"
    commitment = salted_answer_commitment_sha256(original_sha, salt)
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(
            synthetic_contract_v2,
            answer_manifest=manifest,
            revealed_answer_manifest_sha256=original_sha,
            answer_commitment_salt=salt,
            salted_answer_commitment_sha256=commitment,
        )
    )
    assert type(result) is scoring_v2.FormalUnsealedPredictionScoringContractValidationV2


def test_last_answer_row_is_preflighted_before_any_row_or_manifest_hash(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    rows = synthetic_contract_v2.answer.main_answer_rows
    malformed_last = _unchecked_copy(
        rows[-1],
        answer_row_id="not-a-prefixed-answer-row-digest",
    )

    def forbidden_hash(*args: object, **kwargs: object) -> str:
        raise _HostileFieldTouched("late invalid answer row reached hashing")

    for name in (
        "_answer_row_id_v2",
        "_main_row_ids_root_v2",
        "_answer_row_ids_root_v2",
        "_answer_manifest_sha_v2",
    ):
        monkeypatch.setattr(scoring_v2, name, forbidden_hash)
    with pytest.raises(ValueError, match="prefix drift"):
        scoring_v2.build_formal_unsealed_answer_manifest_v2(
            **_builder_kwargs(
                synthetic_contract_v2,
                main_answer_rows=(*rows[:-1], malformed_last),
            )
        )


def test_last_conflict_row_is_preflighted_before_any_partition_root_hash(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    conflict = synthetic_contract_v2.partition.semantic_conflict_row_ids
    malformed_partition = _unchecked_copy(
        synthetic_contract_v2.partition,
        semantic_conflict_row_ids=(*conflict[:-1], "not-a-prefixed-input-row-id"),
    )

    def forbidden_root(*args: object, **kwargs: object) -> str:
        raise _HostileFieldTouched("late invalid conflict row reached root hashing")

    for name in (
        "_main_row_ids_root_v2",
        "_semantic_conflict_row_ids_root_v2",
        "_partition_union_row_ids_root_v2",
        "_partition_manifest_id_v2",
    ):
        monkeypatch.setattr(scoring_v2, name, forbidden_root)
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(
            synthetic_contract_v2,
            partition_manifest=malformed_partition,
        )
    )
    _assert_atomic_rejection(
        result,
        scoring_v2.FormalUnsealedPredictionScoringContractReasonV2
        .PARTITION_MANIFEST_INVALID,
    )


@pytest.mark.parametrize(
    "opening_change",
    (
        {"revealed_answer_manifest_sha256": "not-a-lowercase-sha256"},
        {
            "salted_answer_commitment_sha256": _HostileText(
                f"{980_001:064x}"
            )
        },
        {"answer_commitment_salt": "x" * 4_097},
    ),
)
def test_late_opening_scalar_rejects_before_any_validator_finalization(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_contract_v2: _SyntheticContractFixtureV2,
    opening_change: dict[str, object],
) -> None:
    _forbid_validator_finalization(
        monkeypatch,
        message="invalid late opening scalar reached validator finalization",
    )
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(synthetic_contract_v2, **opening_change)
    )
    _assert_atomic_rejection(
        result,
        scoring_v2.FormalUnsealedPredictionScoringContractReasonV2
        .ANSWER_COMMITMENT_OPENING_INVALID,
    )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        (
            "schema_version",
            "hegel-machine-phase2b-formal-unsealed-answer-manifest/999",
        ),
        (
            "schema_id",
            _hex_id(
                "phase2b_formal_unsealed_answer_manifest_schema_v2_",
                981_001,
            ),
        ),
        (
            "policy_id",
            _hex_id(
                "phase2b_formal_unsealed_answer_manifest_policy_v2_",
                981_002,
            ),
        ),
        ("claim_level", "OTHER_NON_AUTHORITATIVE_CONTRACT_ONLY"),
        ("exact_freeze_id", _hex_id("phase2b_exact_freeze_", 981_003)),
        ("phase2b_protocol_id", _hex_id("phase2b_protocol_", 981_004)),
        (
            "input_archive_version",
            "hegel-machine-phase2b-recognizer-input-archive/999",
        ),
        (
            "input_archive_policy_id",
            _hex_id(
                "phase2b_recognizer_input_archive_policy_v2_",
                981_005,
            ),
        ),
        (
            "batch_policy_id",
            _hex_id("phase2b_trusted_wire_batch_v2_policy_", 981_006),
        ),
    ),
)
def test_wrong_frozen_answer_identity_rejects_before_rows_or_hashes(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_contract_v2: _SyntheticContractFixtureV2,
    field_name: str,
    replacement: str,
) -> None:
    malformed = _unchecked_copy(
        synthetic_contract_v2.answer,
        **{field_name: replacement},
    )
    _forbid_validator_finalization(
        monkeypatch,
        message="wrong frozen answer identity reached validator finalization",
    )

    def forbidden_row_preflight(*args: object, **kwargs: object) -> object:
        raise _HostileFieldTouched(
            "wrong frozen answer identity reached the 720-row preflight"
        )

    monkeypatch.setattr(
        scoring_v2,
        "_preflight_answer_rows_v2",
        forbidden_row_preflight,
    )
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(
            synthetic_contract_v2,
            answer_manifest=malformed,
        )
    )
    _assert_atomic_rejection(
        result,
        scoring_v2.FormalUnsealedPredictionScoringContractReasonV2
        .ANSWER_MANIFEST_INVALID,
    )


def test_validator_last_malformed_answer_row_blocks_all_upstream_finalizers(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_contract_v2: _SyntheticContractFixtureV2,
) -> None:
    rows = synthetic_contract_v2.answer.main_answer_rows
    malformed_last = _unchecked_copy(
        rows[-1],
        answer_row_id="not-a-prefixed-answer-row-digest",
    )
    malformed_answer = _unchecked_copy(
        synthetic_contract_v2.answer,
        main_answer_rows=(*rows[:-1], malformed_last),
    )
    _forbid_validator_finalization(
        monkeypatch,
        message="late malformed answer row reached an upstream finalizer",
    )
    result = scoring_v2.validate_formal_unsealed_prediction_scoring_contract_v2(
        **_validate_kwargs(
            synthetic_contract_v2,
            answer_manifest=malformed_answer,
        )
    )
    _assert_atomic_rejection(
        result,
        scoring_v2.FormalUnsealedPredictionScoringContractReasonV2
        .ANSWER_MANIFEST_INVALID,
    )
