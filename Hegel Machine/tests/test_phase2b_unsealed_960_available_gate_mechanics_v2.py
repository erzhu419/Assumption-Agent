"""Synthetic tests for the non-authoritative available-gate compositor V2.

The positive fixture is constructed data.  It is not an actual recognizer run,
an authenticated reveal, a formal gate decision, effect evidence, or C1 evidence.
"""

from __future__ import annotations

from dataclasses import fields
from enum import Enum
import ast
import hashlib
import inspect
import math
from pathlib import Path
import runpy
from statistics import NormalDist

import pytest

from hegel_machine.hashing import canonical_json, stable_hash
import hegel_machine.phase2b_actual_unsealed_960_replay_input_contract_v2 as input_v2
import hegel_machine.phase2b_formal_unsealed_prediction_scoring_contract_v2 as formal_v2
import hegel_machine.phase2b_recognizer_prediction_archive_v2 as prediction_archive_v2
import hegel_machine.phase2b_unsealed_960_available_gate_mechanics_v2 as gate_v2
import hegel_machine.phase2b_unsealed_960_prediction_scoring_mechanics_v2 as scoring_v2
from hegel_machine.phase2b_freeze_v1 import CanonicalFamilyId
from hegel_machine.phase2b_protocol import Phase2BCaseType
from hegel_machine.phase2b_recognizer_prediction_v2 import PredictionDecisionV2
from hegel_machine.phase2b_wire import RoleBinding


AVAILABLE_RESULT_FIELDS = (
    "metric_name",
    "scope",
    "slice_id",
    "gate_input_definition_id",
    "successes",
    "total",
    "expected_denominator",
    "minimum_point_estimate_ratio",
    "minimum_wilson_lcb_ratio",
    "point_estimate_ratio",
    "point_estimate_hex",
    "one_sided_wilson_lcb_hex",
    "point_threshold_passed",
    "wilson_threshold_passed",
    "available_gate_passed",
    "result_id",
)
UNAVAILABLE_RESULT_FIELDS = (
    "metric_name",
    "scope",
    "gate_input_definition_id",
    "minimum_point_estimate_ratio",
    "minimum_wilson_lcb_ratio",
    "expected_denominator",
    "successes",
    "total",
    "point_estimate_ratio",
    "point_estimate_hex",
    "one_sided_wilson_lcb_hex",
    "point_threshold_passed",
    "wilson_threshold_passed",
    "available_gate_passed",
    "missing_input_reason",
    "unavailable_id",
)
TRUE_CLAIMS = (
    "supplied_scoring_mechanics_graph_independently_verified",
    "supplied_replay_input_contract_graph_independently_verified",
    "supplied_gate_input_manifest_graph_independently_verified",
    "three_supplied_graphs_cross_bound",
    "exact_main_720_row_join_verified",
    "ten_available_overall_gate_mechanics_results_materialized",
    "twenty_four_available_slice_gate_mechanics_results_materialized",
    "frozen_threshold_identity_verified",
    "one_sided_95_percent_wilson_mechanics_evaluated",
    "semantic_conflict_240_excluded_from_available_mechanics",
    "two_unavailable_gate_inputs_retained",
    "atomic_fail_closed_rejection_verified",
)
FALSE_CLAIMS = (
    "formal_gate_evaluation_performed",
    "overall_gate_results_materialized",
    "slice_gate_results_materialized",
    "metric_results_materialized",
    "scored_rows_materialized",
    "formal_wilson_gate_bounds_evaluated",
    "upstream_scoring_control_rejection_metrics_implemented",
    "upstream_scoring_slice_gate_metrics_implemented",
    "scoring_performed",
    "prediction_scored",
    "actual_prediction_scoring_evidence",
    "challenge_in_main_denominator",
    "challenge_scoring_performed",
    "challenge_descriptor_rows_implemented",
    "fail_closed_rate_evaluated",
    "preservation_consistency_evaluated",
    "preservation_evaluated",
    "scale_regret_evaluated",
    "bootstrap_evaluated",
    "fail_closed_gate_inputs_contract_complete",
    "preservation_gate_inputs_contract_complete",
    "scale_regret_inputs_contract_complete",
    "bootstrap_inputs_contract_complete",
    "baseline_outputs_verified",
    "margin_stratum_authority_verified",
    "family_slice_label_authority_verified",
    "scale_slice_semantics_authority_verified",
    "latent_case_independence_verified",
    "raw_input_archive_replayed",
    "raw_prediction_archive_replayed",
    "answer_commitment_opening_verified",
    "gate_input_commitment_opening_verified",
    "prediction_commit_before_reveal_verified",
    "evidence_supplied",
    "evidence_verified",
    "answer_manifest_authority_verified",
    "gate_input_manifest_authority_verified",
    "answer_commitment_authority_verified",
    "gate_input_commitment_authority_verified",
    "pre_reveal_commitment_timing_verified",
    "one_shot_policy_enforced",
    "durable_attempt_ledger_verified",
    "secret_custodian_replay_verified",
    "input_archive_membership_verified",
    "batch_policy_membership_verified",
    "source_registry_projection_verified",
    "source_public_disjoint_verified",
    "single_live_allocation_verified",
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
    "effect_evidence",
    "formal_c1_report_verified",
    "c1_exit_evidence",
)
SUCCESS_IDENTITY_FIELDS = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "result_id",
    "scoring_mechanics_result_id",
    "replay_input_contract_result_id",
    "gate_input_manifest_id",
    "gate_input_manifest_sha256",
    "gate_input_manifest_schema_id",
    "gate_input_manifest_policy_id",
    "prediction_archive_id",
    "prediction_archive_sha256",
    "answer_manifest_id",
    "answer_manifest_sha256",
    "scoring_mechanics_schema_id",
    "scoring_mechanics_policy_id",
    "scoring_mechanics_version",
    "scoring_mechanics_claim_level",
    "replay_input_contract_schema_id",
    "replay_input_contract_policy_id",
    "replay_input_contract_version",
    "replay_input_contract_claim_level",
    "gate_input_manifest_schema_version",
    "gate_input_manifest_claim_level",
    "protocol_id",
    "formal_scoring_contract_id",
    "formal_scoring_contract_schema_id",
    "formal_scoring_contract_policy_id",
    "formal_scoring_contract_version",
    "formal_scoring_contract_claim_level",
    "main_row_ids_root",
    "semantic_conflict_row_ids_root",
    "main_answer_row_ids_root",
    "main_gate_input_row_ids_root",
    "main_row_count",
    "semantic_conflict_excluded_count",
    "overall_result_count",
    "slice_result_count",
    "unavailable_result_count",
)
SUCCESS_FIELDS = (
    *SUCCESS_IDENTITY_FIELDS,
    *TRUE_CLAIMS,
    *FALSE_CLAIMS,
    "available_overall_gate_mechanics_results",
    "available_slice_gate_mechanics_results",
    "unavailable_gate_mechanics",
)
REJECTION_FIELDS = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "validation",
    "available_overall_gate_mechanics_results",
    "available_slice_gate_mechanics_results",
    "unavailable_gate_mechanics",
    "partial_output_published",
    *TRUE_CLAIMS,
    *FALSE_CLAIMS,
)

ANSWER_MANIFEST_SCHEMA_VERSION = (
    "hegel-machine-phase2b-formal-unsealed-answer-manifest/2"
)
ANSWER_MANIFEST_SCHEMA_ID = (
    "phase2b_formal_unsealed_answer_manifest_schema_v2_"
    "3f427810029665a54854751b7d021a77c4d5f874b7df1992d50434b7108d32f0"
)
ANSWER_MANIFEST_POLICY_ID = (
    "phase2b_formal_unsealed_answer_manifest_policy_v2_"
    "be684716aadb4bb6cced67348233d0c6ca78d7e0c98c6df2542bcc1787c50f1e"
)
ANSWER_ROW_DOMAIN = b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_ROW/V2\x00"
ANSWER_MANIFEST_DOMAIN = (
    b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_MANIFEST/V2\x00"
)
ANSWER_ROW_PREFIX = "phase2b_formal_unsealed_answer_row_v2_"
ANSWER_MANIFEST_PREFIX = "phase2b_formal_unsealed_answer_manifest_v2_"

SCORING_ADDRESS_PREFIXES = {
    "schema_id": "phase2b_unsealed_960_prediction_scoring_mechanics_schema_v2_",
    "policy_id": "phase2b_unsealed_960_prediction_scoring_mechanics_policy_v2_",
    "result_id": "phase2b_unsealed_960_prediction_scoring_mechanics_v2_",
    "prediction_archive_id": "phase2b_recognizer_prediction_archive_v2_",
    "prediction_archive_policy_id": "phase2b_recognizer_prediction_archive_policy_v2_",
    "run_context_id": "phase2b_public_prediction_run_context_v2_",
    "input_archive_id": "phase2b_recognizer_input_archive_v2_",
    "batch_id": "phase2b_trusted_wire_batch_v2_",
    "execution_freeze_manifest_id": "phase2b_execution_freeze_",
    "protocol_id": "phase2b_protocol_",
    "structural_receipt_id": "phase2b_strict_recognizer_receipt_v2_",
    "partition_manifest_id": "phase2b_unsealed_prediction_partition_v2_",
    "answer_manifest_id": "phase2b_formal_unsealed_answer_manifest_v2_",
    "formal_scoring_contract_id": (
        "phase2b_formal_unsealed_prediction_scoring_contract_v2_"
    ),
    "ordered_archive_input_row_ids_root": "phase2b_prediction_input_rows_v2_",
    "main_row_ids_root": "phase2b_unsealed_main_rows_v2_",
    "semantic_conflict_row_ids_root": (
        "phase2b_unsealed_semantic_conflict_rows_v2_"
    ),
    "partition_union_row_ids_root": (
        "phase2b_unsealed_partition_union_rows_v2_"
    ),
    "main_answer_row_ids_root": "phase2b_formal_unsealed_answer_rows_v2_",
}
REPLAY_ADDRESS_PREFIXES = {
    "schema_id": "phase2b_actual_unsealed_960_replay_input_contract_schema_v2_",
    "policy_id": "phase2b_actual_unsealed_960_replay_input_contract_policy_v2_",
    "result_id": "phase2b_actual_unsealed_960_replay_input_contract_v2_",
    "gate_input_manifest_id": "phase2b_formal_unsealed_gate_input_manifest_v2_",
    "gate_input_manifest_schema_id": (
        "phase2b_formal_unsealed_gate_input_manifest_schema_v2_"
    ),
    "gate_input_manifest_policy_id": (
        "phase2b_formal_unsealed_gate_input_manifest_policy_v2_"
    ),
    "answer_manifest_id": "phase2b_formal_unsealed_answer_manifest_v2_",
    "execution_freeze_manifest_id": "phase2b_execution_freeze_",
    "input_archive_id": "phase2b_recognizer_input_archive_v2_",
    "input_archive_policy_id": "phase2b_recognizer_input_archive_policy_v2_",
    "batch_id": "phase2b_trusted_wire_batch_v2_",
    "batch_policy_id": "phase2b_trusted_wire_batch_v2_policy_",
    "exact_freeze_id": "phase2b_exact_freeze_",
    "protocol_id": "phase2b_protocol_",
    "formal_scoring_contract_id": (
        "phase2b_formal_unsealed_prediction_scoring_contract_v2_"
    ),
    "ordered_archive_input_row_ids_root": "phase2b_prediction_input_rows_v2_",
    "main_row_ids_root": "phase2b_unsealed_main_rows_v2_",
    "semantic_conflict_row_ids_root": (
        "phase2b_unsealed_semantic_conflict_rows_v2_"
    ),
    "partition_union_row_ids_root": (
        "phase2b_unsealed_partition_union_rows_v2_"
    ),
    "main_answer_row_ids_root": "phase2b_formal_unsealed_answer_rows_v2_",
    "main_gate_input_row_ids_root": (
        "phase2b_actual_replay_gate_input_rows_v2_"
    ),
}
MANIFEST_ADDRESS_PREFIXES = {
    "schema_id": "phase2b_formal_unsealed_gate_input_manifest_schema_v2_",
    "policy_id": "phase2b_formal_unsealed_gate_input_manifest_policy_v2_",
    "exact_freeze_id": "phase2b_exact_freeze_",
    "phase2b_protocol_id": "phase2b_protocol_",
    "formal_scoring_contract_id": (
        "phase2b_formal_unsealed_prediction_scoring_contract_v2_"
    ),
    "execution_freeze_manifest_id": "phase2b_execution_freeze_",
    "input_archive_id": "phase2b_recognizer_input_archive_v2_",
    "input_archive_policy_id": "phase2b_recognizer_input_archive_policy_v2_",
    "batch_id": "phase2b_trusted_wire_batch_v2_",
    "batch_policy_id": "phase2b_trusted_wire_batch_v2_policy_",
    "ordered_archive_input_row_ids_root": "phase2b_prediction_input_rows_v2_",
    "main_row_ids_root": "phase2b_unsealed_main_rows_v2_",
    "semantic_conflict_row_ids_root": (
        "phase2b_unsealed_semantic_conflict_rows_v2_"
    ),
    "partition_union_row_ids_root": (
        "phase2b_unsealed_partition_union_rows_v2_"
    ),
    "answer_manifest_id": "phase2b_formal_unsealed_answer_manifest_v2_",
    "main_answer_row_ids_root": "phase2b_formal_unsealed_answer_rows_v2_",
    "main_gate_input_row_ids_root": (
        "phase2b_actual_replay_gate_input_rows_v2_"
    ),
    "gate_input_manifest_id": "phase2b_formal_unsealed_gate_input_manifest_v2_",
}

OVERALL_THRESHOLDS = (
    ("family_exact", 240, (90, 100), (86, 100)),
    ("binding_exact", 240, (90, 100), (86, 100)),
    ("scale_set_accuracy", 240, (87, 100), (82, 100)),
    ("joint_exact", 240, (85, 100), (80, 100)),
    ("hard_negative_rejection", 96, (95, 100), (90, 100)),
    ("binding_counterfactual_rejection", 96, (95, 100), (90, 100)),
    ("scale_counterfactual_rejection", 96, (93, 100), (88, 100)),
    ("sign_or_invariant_break_rejection", 96, (95, 100), (90, 100)),
    ("abstention_specificity", 228, (95, 100), (90, 100)),
    ("nonidentifiable_scale_abstention", 96, (95, 100), (90, 100)),
)
SLICE_THRESHOLDS = (
    ("answerable_joint_exact", "family", 40, (80, 100), (70, 100)),
    ("all_control_rejection", "family", 64, (88, 100), (78, 100)),
    ("abstention_specificity", "family", 38, (85, 100), (75, 100)),
    ("answerable_joint_exact", "scale", 120, (80, 100), (70, 100)),
    ("all_control_rejection", "scale", 192, (88, 100), (78, 100)),
    ("abstention_specificity", "scale", 114, (85, 100), (75, 100)),
)
CONTROL_TYPES = frozenset(
    {
        Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE,
        Phase2BCaseType.BINDING_COUNTERFACTUAL,
        Phase2BCaseType.SCALE_COUNTERFACTUAL,
        Phase2BCaseType.SIGN_OR_INVARIANT_BREAK,
    }
)
ANSWERABLE_TYPES = frozenset(
    {
        Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE,
        Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE,
    }
)
SCORING_TRUE_CLAIMS = (
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
SCORING_FALSE_CLAIMS = (
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


class _PrehashBoundaryReached(BaseException):
    """Sentinel proving malformed input reached hashing or Wilson work."""


class _HostileEqualityReached(BaseException):
    """Sentinel proving a required scalar was compared before exact typing."""


class _HostileEquality:
    def __init__(self) -> None:
        self.comparison_attempts = 0

    def __eq__(self, other: object) -> bool:
        self.comparison_attempts += 1
        raise _HostileEqualityReached("hostile equality was invoked")

    def __ne__(self, other: object) -> bool:
        self.comparison_attempts += 1
        raise _HostileEqualityReached("hostile inequality was invoked")


class _FakeCaseType(str, Enum):
    UNIQUE_SCALE_ANSWERABLE = Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE.value


class _FakeDecision(str, Enum):
    ANSWER = PredictionDecisionV2.ANSWER.value


class _FakeFamily(str, Enum):
    F01 = CanonicalFamilyId.F01.value


def _issue(value_type: type[object], values: dict[str, object]) -> object:
    value = object.__new__(value_type)
    assert set(values) == {item.name for item in fields(value_type)}
    for name, item in values.items():
        object.__setattr__(value, name, item)
    return value


def _unchecked_copy(value: object, **changes: object) -> object:
    return _issue(
        type(value),
        {
            item.name: changes.get(
                item.name, object.__getattribute__(value, item.name)
            )
            for item in fields(value)
        },
    )


def _plain(value: object) -> object:
    if type(value) in (str, int, bool) or value is None:
        return value
    if isinstance(value, Enum):
        return value.value
    if type(value) is RoleBinding:
        return {"role_id": value.role_id, "entity_id": value.entity_id}
    if type(value) is tuple:
        return [_plain(item) for item in value]
    if hasattr(type(value), "__dataclass_fields__"):
        return {
            item.name: _plain(object.__getattribute__(value, item.name))
            for item in fields(value)
        }
    raise TypeError(f"unsupported independent primitive: {type(value)!r}")


def _scoring_id(value: object, fields_without_id: tuple[str, ...], prefix: str) -> str:
    return stable_hash(
        {
            name: _plain(object.__getattribute__(value, name))
            for name in fields_without_id
        },
        prefix=prefix,
    )


def _wilson(successes: int, total: int) -> float:
    proportion = successes / total
    z = NormalDist().inv_cdf(0.95)
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = proportion + z2 / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total)
    )
    return max(0.0, (center - radius) / denominator)


def test_independent_math_sqrt_wilson_hex_and_threshold_boundary_oracle() -> None:
    assert _wilson(213, 240).hex() == "0x1.b2f6ef1be437ap-1"
    assert _wilson(183, 240).hex() == "0x1.6dd9475c10aaap-1"
    assert _wilson(84, 96).hex() == "0x1.9e36a7deeb269p-1"
    assert _wilson(216, 228).hex() == "0x1.d5b293b5fbdccp-1"
    below = _wilson(215, 240)
    at_or_above = _wilson(216, 240)
    assert below.hex() == "0x1.b7bd826257e62p-1"
    assert at_or_above.hex() == "0x1.ba23c9bea73ecp-1"
    assert below < 86 / 100 <= at_or_above


def _ratio(value: tuple[int, int]) -> str:
    divisor = math.gcd(*value)
    return f"{value[0] // divisor}/{value[1] // divisor}"


def _domain_id(preimage: object, *, domain: bytes, prefix: str) -> str:
    return prefix + hashlib.sha256(
        domain + canonical_json(preimage).encode("utf-8")
    ).hexdigest()


def _sequence_root(values: tuple[str, ...], *, domain: bytes, prefix: str) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(len(values).to_bytes(4, "big"))
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return prefix + digest.hexdigest()


def _rehash_scoring(
    value: scoring_v2.Unsealed960PredictionScoringMechanicsV2,
    **changes: object,
) -> scoring_v2.Unsealed960PredictionScoringMechanicsV2:
    changed = _unchecked_copy(value, result_id="", **changes)
    object.__setattr__(
        changed,
        "result_id",
        _scoring_id(
            changed,
            tuple(
                item.name
                for item in fields(type(changed))
                if item.name != "result_id"
            ),
            "phase2b_unsealed_960_prediction_scoring_mechanics_v2_",
        ),
    )
    return changed  # type: ignore[return-value]


def _rehash_row(
    value: scoring_v2.Unsealed960MainRowResultV2,
    **changes: object,
) -> scoring_v2.Unsealed960MainRowResultV2:
    changed = _unchecked_copy(value, row_result_id="", **changes)
    object.__setattr__(
        changed,
        "row_result_id",
        _scoring_id(
            changed,
            tuple(item.name for item in fields(type(changed)))[:-1],
            "phase2b_unsealed_960_main_row_result_v2_",
        ),
    )
    return changed  # type: ignore[return-value]


def _rehash_metric(
    value: scoring_v2.Unsealed960MetricResultV2,
    **changes: object,
) -> scoring_v2.Unsealed960MetricResultV2:
    changed = _unchecked_copy(value, metric_result_id="", **changes)
    object.__setattr__(
        changed,
        "metric_result_id",
        _scoring_id(
            changed,
            tuple(item.name for item in fields(type(changed)))[:-1],
            "phase2b_unsealed_960_metric_result_v2_",
        ),
    )
    return changed  # type: ignore[return-value]


def _rehash_replay(
    value: input_v2.ActualUnsealed960ReplayInputContractV2,
    **changes: object,
) -> input_v2.ActualUnsealed960ReplayInputContractV2:
    changed = _unchecked_copy(value, result_id="", **changes)
    preimage = {
        item.name: _plain(object.__getattribute__(changed, item.name))
        for item in fields(type(changed))
        if item.name != "result_id"
    }
    object.__setattr__(
        changed,
        "result_id",
        _domain_id(
            preimage,
            domain=b"HEGEL/PHASE2B/ACTUAL_REPLAY/RESULT/V2\x00",
            prefix="phase2b_actual_unsealed_960_replay_input_contract_v2_",
        ),
    )
    return changed  # type: ignore[return-value]


def _gate_row_mapping(value: input_v2.FormalUnsealedGateInputRowV2) -> dict[str, object]:
    return {
        "input_row_id": value.input_row_id,
        "answer_row_id": value.answer_row_id,
        "case_type": value.case_type.value,
        "margin_stratum": value.margin_stratum.value,
        "canonical_family_id": value.canonical_family_id.value,
        "scale_slice_id": value.scale_slice_id.value,
        "latent_base_case_id": value.latent_base_case_id,
    }


def _rehash_gate_row(
    value: input_v2.FormalUnsealedGateInputRowV2,
    **changes: object,
) -> input_v2.FormalUnsealedGateInputRowV2:
    changed = _unchecked_copy(value, gate_input_row_id="", **changes)
    object.__setattr__(
        changed,
        "gate_input_row_id",
        _domain_id(
            _gate_row_mapping(changed),  # type: ignore[arg-type]
            domain=b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_ROW/V2\x00",
            prefix="phase2b_actual_replay_gate_input_row_v2_",
        ),
    )
    return changed  # type: ignore[return-value]


def _rehash_gate_manifest(
    value: input_v2.FormalUnsealedGateInputManifestV2,
    **changes: object,
) -> input_v2.FormalUnsealedGateInputManifestV2:
    changed = _unchecked_copy(
        value,
        gate_input_manifest_sha256="",
        gate_input_manifest_id="",
        **changes,
    )
    preimage = {
        "schema_version": changed.schema_version,
        "schema_id": changed.schema_id,
        "policy_id": changed.policy_id,
        "claim_level": changed.claim_level,
        "exact_freeze_id": changed.exact_freeze_id,
        "phase2b_protocol_id": changed.phase2b_protocol_id,
        "formal_scoring_contract_id": changed.formal_scoring_contract_id,
        "execution_freeze_manifest_id": changed.execution_freeze_manifest_id,
        "input_archive_id": changed.input_archive_id,
        "input_archive_sha256": changed.input_archive_sha256,
        "input_archive_version": changed.input_archive_version,
        "input_archive_policy_id": changed.input_archive_policy_id,
        "batch_id": changed.batch_id,
        "batch_policy_id": changed.batch_policy_id,
        "ordered_archive_input_row_ids_root": changed.ordered_archive_input_row_ids_root,
        "main_row_ids_root": changed.main_row_ids_root,
        "semantic_conflict_row_ids_root": changed.semantic_conflict_row_ids_root,
        "partition_union_row_ids_root": changed.partition_union_row_ids_root,
        "answer_manifest_id": changed.answer_manifest_id,
        "answer_manifest_sha256": changed.answer_manifest_sha256,
        "main_answer_row_ids_root": changed.main_answer_row_ids_root,
        "main_gate_input_rows": [
            {**_gate_row_mapping(item), "gate_input_row_id": item.gate_input_row_id}
            for item in changed.main_gate_input_rows
        ],
        "main_gate_input_row_ids_root": changed.main_gate_input_row_ids_root,
        "required_evidence_inventory": [
            _plain(item) for item in changed.required_evidence_inventory
        ],
    }
    manifest_sha = hashlib.sha256(
        b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_MANIFEST/V2\x00"
        + canonical_json(preimage).encode("utf-8")
    ).hexdigest()
    object.__setattr__(changed, "gate_input_manifest_sha256", manifest_sha)
    object.__setattr__(
        changed,
        "gate_input_manifest_id",
        "phase2b_formal_unsealed_gate_input_manifest_v2_" + manifest_sha,
    )
    return changed  # type: ignore[return-value]


def _answer_row_mapping(
    value: scoring_v2.Unsealed960MainRowResultV2,
) -> dict[str, object]:
    return {
        "input_row_id": value.input_row_id,
        "case_type": value.case_type.value,
        "expected_decision": value.expected_decision.value,
        "canonical_family_id": (
            None
            if value.expected_canonical_family_id is None
            else value.expected_canonical_family_id.value
        ),
        "binding": [_plain(item) for item in value.expected_binding],
        "admissible_scale_ids": list(value.expected_admissible_scale_ids),
        "answer_row_id": value.answer_row_id,
    }


def _answer_manifest_identity(
    rows: tuple[scoring_v2.Unsealed960MainRowResultV2, ...],
    replay: input_v2.ActualUnsealed960ReplayInputContractV2,
) -> tuple[str, str, str]:
    row_mappings = [_answer_row_mapping(item) for item in rows]
    answer_root = _sequence_root(
        tuple(item["answer_row_id"] for item in row_mappings),  # type: ignore[misc]
        domain=b"HEGEL/PHASE2B/FORMAL_UNSEALED/ANSWER_ROW_IDS/V2\x00",
        prefix="phase2b_formal_unsealed_answer_rows_v2_",
    )
    preimage = {
        "schema_version": ANSWER_MANIFEST_SCHEMA_VERSION,
        "schema_id": ANSWER_MANIFEST_SCHEMA_ID,
        "policy_id": ANSWER_MANIFEST_POLICY_ID,
        "claim_level": formal_v2.FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL,
        "exact_freeze_id": replay.exact_freeze_id,
        "phase2b_protocol_id": replay.protocol_id,
        "execution_freeze_manifest_id": replay.execution_freeze_manifest_id,
        "input_archive_id": replay.input_archive_id,
        "input_archive_sha256": replay.input_archive_sha256,
        "input_archive_version": replay.input_archive_version,
        "input_archive_policy_id": replay.input_archive_policy_id,
        "batch_id": replay.batch_id,
        "batch_policy_id": replay.batch_policy_id,
        "ordered_archive_input_row_ids_root": replay.ordered_archive_input_row_ids_root,
        "main_row_ids_root": replay.main_row_ids_root,
        "semantic_conflict_row_ids_root": replay.semantic_conflict_row_ids_root,
        "partition_union_row_ids_root": replay.partition_union_row_ids_root,
        "main_answer_rows": row_mappings,
        "main_answer_row_ids_root": answer_root,
    }
    answer_sha = hashlib.sha256(
        ANSWER_MANIFEST_DOMAIN + canonical_json(preimage).encode("utf-8")
    ).hexdigest()
    return ANSWER_MANIFEST_PREFIX + answer_sha, answer_sha, answer_root


def _different_family(value: CanonicalFamilyId) -> CanonicalFamilyId:
    return next(item for item in CanonicalFamilyId if item is not value)


def _different_binding(index: int) -> tuple[RoleBinding, ...]:
    return (
        RoleBinding(
            role_id=f"00000000-0000-4000-8000-{700_000 + index:012x}",
            entity_id=f"00000000-0000-4000-8000-{800_000 + index:012x}",
        ),
    )


def _different_scales(index: int, count: int) -> tuple[str, ...]:
    return tuple(
        f"00000000-0000-4000-8000-{900_000 + index * 4 + offset:012x}"
        for offset in range(count)
    )


def _metric_success(
    metric_name: str,
    *,
    case_type: Phase2BCaseType,
    predicted_decision: PredictionDecisionV2,
    expected_decision: PredictionDecisionV2,
    family_exact: bool,
    binding_exact: bool,
    scale_exact: bool,
) -> tuple[bool, bool | None]:
    answerable = case_type in ANSWERABLE_TYPES
    if metric_name == "answerable_count":
        return answerable, None
    if metric_name == "family_exact_accuracy":
        return answerable, family_exact if answerable else None
    if metric_name == "binding_exact_accuracy":
        return answerable, binding_exact if answerable else None
    if metric_name == "scale_set_accuracy":
        return answerable, scale_exact if answerable else None
    if metric_name == "unique_scale_accuracy":
        eligible = case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE
        return eligible, (
            predicted_decision is PredictionDecisionV2.ANSWER and scale_exact
        ) if eligible else None
    if metric_name == "joint_exact_accuracy":
        return answerable, (
            predicted_decision is expected_decision
            and family_exact
            and binding_exact
            and scale_exact
        ) if answerable else None
    if metric_name == "abstention_specificity":
        eligible = case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE
        return eligible, (
            predicted_decision is not PredictionDecisionV2.ABSTAIN
        ) if eligible else None
    if metric_name == "nonidentifiability_abstention_accuracy":
        eligible = case_type is Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE
        return eligible, (
            predicted_decision is PredictionDecisionV2.ABSTAIN
        ) if eligible else None
    if metric_name == "set_valued_answer_accuracy":
        eligible = case_type is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE
        return eligible, (
            predicted_decision is PredictionDecisionV2.ANSWER_SET
            and family_exact
            and binding_exact
            and scale_exact
        ) if eligible else None
    raise AssertionError(metric_name)


def _prediction_for_row(
    *,
    index: int,
    cell_index: int,
    case_ordinal: int,
    answer: formal_v2.FormalUnsealedAnswerRowV2,
    family: CanonicalFamilyId,
) -> tuple[
    PredictionDecisionV2,
    CanonicalFamilyId | None,
    tuple[RoleBinding, ...],
    tuple[str, ...],
]:
    if answer.case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE:
        if case_ordinal == 18:
            return PredictionDecisionV2.ABSTAIN, None, (), ()
        predicted_family = answer.canonical_family_id
        predicted_binding = answer.binding
        predicted_scales = answer.admissible_scale_ids
        if case_ordinal == 15:
            assert predicted_family is not None
            predicted_family = _different_family(predicted_family)
        elif case_ordinal == 16:
            predicted_binding = _different_binding(index)
        elif case_ordinal == 17:
            predicted_scales = _different_scales(index, 1)
        return (
            PredictionDecisionV2.ANSWER,
            predicted_family,
            predicted_binding,
            predicted_scales,
        )
    if answer.case_type is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE:
        predicted_family = answer.canonical_family_id
        predicted_binding = answer.binding
        predicted_scales = answer.admissible_scale_ids
        mode = cell_index % 4
        if mode == 1:
            assert predicted_family is not None
            predicted_family = _different_family(predicted_family)
        elif mode == 2:
            predicted_binding = _different_binding(index)
        elif mode == 3:
            predicted_scales = _different_scales(index, 2)
        return (
            PredictionDecisionV2.ANSWER_SET,
            predicted_family,
            predicted_binding,
            predicted_scales,
        )
    if answer.case_type in CONTROL_TYPES or answer.case_type is (
        Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE
    ):
        if case_ordinal < 7:
            return PredictionDecisionV2.ABSTAIN, None, (), ()
        return (
            PredictionDecisionV2.ANSWER,
            family,
            _different_binding(index),
            _different_scales(index, 1),
        )
    raise AssertionError(answer.case_type)


@pytest.fixture(scope="module")
def supplied_graphs() -> tuple[
    scoring_v2.Unsealed960PredictionScoringMechanicsV2,
    input_v2.ActualUnsealed960ReplayInputContractV2,
    input_v2.FormalUnsealedGateInputManifestV2,
]:
    namespace = runpy.run_path(
        str(Path(__file__).with_name(
            "test_phase2b_actual_unsealed_960_replay_input_contract_v2.py"
        ))
    )
    fixture = namespace["_synthetic_fixture_base"]()
    gate_rows = namespace["_gate_rows"](fixture)
    manifest = input_v2.build_formal_unsealed_gate_input_manifest_v2(
        answer_manifest=fixture.answer,
        main_gate_input_rows=gate_rows,
    )
    salt = "synthetic-available-gate-only-salt-0123456789abcdef"
    commitment = input_v2.salted_gate_input_commitment_sha256_v2(
        manifest.gate_input_manifest_sha256,
        salt,
    )
    replay = input_v2.validate_actual_unsealed_960_replay_input_contract_v2(
        gate_input_manifest=manifest,
        answer_manifest=fixture.answer,
        revealed_gate_input_manifest_sha256=manifest.gate_input_manifest_sha256,
        gate_input_commitment_salt=salt,
        salted_gate_input_commitment_sha256=commitment,
    )
    assert type(replay) is input_v2.ActualUnsealed960ReplayInputContractV2

    definitions = formal_v2.frozen_formal_unsealed_prediction_scoring_contract_v2().metric_definitions
    rows: list[scoring_v2.Unsealed960MainRowResultV2] = []
    case_ordinals = {item: 0 for item in Phase2BCaseType}
    for index, (answer, gate_row) in enumerate(
        zip(fixture.answer.main_answer_rows, manifest.main_gate_input_rows, strict=True)
    ):
        cell_index = index // 60
        case_ordinal = case_ordinals[answer.case_type] % (
            19 if answer.case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE else 8
        )
        if answer.case_type is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE:
            case_ordinal = 0
        case_ordinals[answer.case_type] += 1
        decision, predicted_family, predicted_binding, predicted_scales = _prediction_for_row(
            index=index,
            cell_index=cell_index,
            case_ordinal=case_ordinal,
            answer=answer,
            family=gate_row.canonical_family_id,
        )
        positive = decision in {
            PredictionDecisionV2.ANSWER,
            PredictionDecisionV2.ANSWER_SET,
        }
        family_exact = positive and predicted_family is answer.canonical_family_id
        binding_exact = positive and predicted_binding == answer.binding
        scale_exact = positive and predicted_scales == answer.admissible_scale_ids
        outcomes: list[scoring_v2.Unsealed960MetricRowOutcomeV2] = []
        for definition in definitions:
            eligible, success = _metric_success(
                definition.metric_name,
                case_type=answer.case_type,
                predicted_decision=decision,
                expected_decision=answer.expected_decision,
                family_exact=family_exact,
                binding_exact=binding_exact,
                scale_exact=scale_exact,
            )
            outcome = _issue(
                scoring_v2.Unsealed960MetricRowOutcomeV2,
                {
                    "metric_definition_id": definition.metric_definition_id,
                    "metric_name": definition.metric_name,
                    "eligible": eligible,
                    "success": success,
                    "metric_row_outcome_id": "",
                },
            )
            object.__setattr__(
                outcome,
                "metric_row_outcome_id",
                _scoring_id(
                    outcome,
                    tuple(item.name for item in fields(type(outcome)))[:-1],
                    "phase2b_unsealed_960_metric_row_outcome_v2_",
                ),
            )
            outcomes.append(outcome)  # type: ignore[arg-type]
        answerable = answer.case_type in ANSWERABLE_TYPES
        row = _issue(
            scoring_v2.Unsealed960MainRowResultV2,
            {
                "input_row_id": answer.input_row_id,
                "prediction_record_id": "phase2b_recognizer_prediction_record_v2_" + hashlib.sha256(f"record-{index}".encode()).hexdigest(),
                "prediction_content_id": "phase2b_prediction_" + hashlib.sha256(f"prediction-{index}".encode()).hexdigest(),
                "answer_row_id": answer.answer_row_id,
                "case_type": answer.case_type,
                "predicted_decision": decision,
                "expected_decision": answer.expected_decision,
                "predicted_canonical_family_id": predicted_family,
                "expected_canonical_family_id": answer.canonical_family_id,
                "predicted_binding": predicted_binding,
                "expected_binding": answer.binding,
                "predicted_admissible_scale_ids": predicted_scales,
                "expected_admissible_scale_ids": answer.admissible_scale_ids,
                "decision_exact": decision is answer.expected_decision,
                "family_exact": family_exact if answerable else None,
                "binding_exact": binding_exact if answerable else None,
                "scale_set_exact": scale_exact if answerable else None,
                "joint_exact": (
                    decision is answer.expected_decision
                    and family_exact
                    and binding_exact
                    and scale_exact
                ) if answerable else None,
                "metric_eligible": answer.case_type in ANSWERABLE_TYPES or answer.case_type is Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE,
                "metric_outcomes": tuple(outcomes),
                "row_result_id": "",
            },
        )
        object.__setattr__(
            row,
            "row_result_id",
            _scoring_id(
                row,
                tuple(item.name for item in fields(type(row)))[:-1],
                "phase2b_unsealed_960_main_row_result_v2_",
            ),
        )
        rows.append(row)  # type: ignore[arg-type]

    metric_results: list[scoring_v2.Unsealed960MetricResultV2] = []
    for definition in definitions:
        outcomes = tuple(
            outcome
            for row in rows
            for outcome in row.metric_outcomes
            if outcome.metric_definition_id == definition.metric_definition_id
        )
        eligible = tuple(item for item in outcomes if item.eligible)
        result = _issue(
            scoring_v2.Unsealed960MetricResultV2,
            {
                "metric_definition_id": definition.metric_definition_id,
                "metric_name": definition.metric_name,
                "metric_kind": definition.metric_kind,
                "denominator_case_types": definition.denominator_case_types,
                "expected_denominator": definition.expected_denominator,
                "observed_denominator": len(eligible),
                "success_count": None if definition.metric_kind is formal_v2.FormalUnsealedMetricKindV2.COUNT else sum(item.success is True for item in eligible),
                "count_value": len(eligible) if definition.metric_kind is formal_v2.FormalUnsealedMetricKindV2.COUNT else None,
                "success_rule": definition.success_rule,
                "separately_reported": definition.separately_reported,
                "metric_result_id": "",
            },
        )
        object.__setattr__(
            result,
            "metric_result_id",
            _scoring_id(
                result,
                tuple(item.name for item in fields(type(result)))[:-1],
                "phase2b_unsealed_960_metric_result_v2_",
            ),
        )
        metric_results.append(result)  # type: ignore[arg-type]

    mechanics_values: dict[str, object] = {
        "disposition": scoring_v2.Unsealed960PredictionScoringDispositionV2.MECHANICS_COMPLETE_NOT_ACTUAL_EXECUTION,
        "reason": scoring_v2.Unsealed960PredictionScoringReasonV2.CANONICAL_V2_MAIN_ROW_NINE_METRIC_MECHANICS_COMPLETE,
        "version": scoring_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_VERSION,
        "schema_id": scoring_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_SCHEMA_ID,
        "policy_id": scoring_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_POLICY_ID,
        "claim_level": scoring_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_CLAIM_LEVEL,
        "result_id": "",
        "prediction_archive_id": "phase2b_recognizer_prediction_archive_v2_" + hashlib.sha256(b"synthetic available gate archive").hexdigest(),
        "prediction_archive_sha256": hashlib.sha256(b"synthetic available gate archive bytes").hexdigest(),
        "prediction_archive_version": "hegel-machine-phase2b-recognizer-prediction-archive/2",
        "prediction_archive_policy_id": (
            prediction_archive_v2.RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2
        ),
        "run_context_id": "phase2b_public_prediction_run_context_v2_" + "2" * 64,
        "input_archive_id": replay.input_archive_id,
        "input_archive_sha256": replay.input_archive_sha256,
        "batch_id": replay.batch_id,
        "execution_freeze_manifest_id": replay.execution_freeze_manifest_id,
        "protocol_id": replay.protocol_id,
        "structural_receipt_id": "phase2b_strict_recognizer_receipt_v2_" + "3" * 64,
        "partition_manifest_id": "phase2b_unsealed_prediction_partition_v2_" + "4" * 64,
        "answer_manifest_id": replay.answer_manifest_id,
        "answer_manifest_sha256": replay.answer_manifest_sha256,
        "salted_answer_commitment_sha256": hashlib.sha256(b"synthetic answer commitment").hexdigest(),
        "formal_scoring_contract_id": replay.formal_scoring_contract_id,
        "ordered_archive_input_row_ids_root": replay.ordered_archive_input_row_ids_root,
        "main_row_ids_root": replay.main_row_ids_root,
        "semantic_conflict_row_ids_root": replay.semantic_conflict_row_ids_root,
        "partition_union_row_ids_root": replay.partition_union_row_ids_root,
        "main_answer_row_ids_root": replay.main_answer_row_ids_root,
        "total_prediction_count": 960,
        "main_row_result_count": 720,
        "metric_eligible_main_row_count": 336,
        "control_row_without_frozen_metric_count": 384,
        "semantic_conflict_excluded_count": 240,
        "metric_results": tuple(metric_results),
        "main_row_results": tuple(rows),
        "gate_results": (),
        "scale_regret_result": None,
        "bootstrap_result": None,
    }
    mechanics_values.update({name: True for name in SCORING_TRUE_CLAIMS})
    mechanics_values.update({name: False for name in SCORING_FALSE_CLAIMS})
    mechanics = _issue(
        scoring_v2.Unsealed960PredictionScoringMechanicsV2,
        mechanics_values,
    )
    object.__setattr__(
        mechanics,
        "result_id",
        _scoring_id(
            mechanics,
            tuple(
                item.name
                for item in fields(type(mechanics))
                if item.name != "result_id"
            ),
            "phase2b_unsealed_960_prediction_scoring_mechanics_v2_",
        ),
    )
    return mechanics, replay, manifest  # type: ignore[return-value]


def _evaluate_kwargs(
    graphs: tuple[object, object, object], **changes: object
) -> dict[str, object]:
    values = dict(
        zip(
            ("scoring_mechanics", "replay_input_contract", "gate_input_manifest"),
            graphs,
            strict=True,
        )
    )
    values.update(changes)
    return values


def _assert_atomic_rejection(
    value: object,
    reason: gate_v2.Unsealed960AvailableGateMechanicsReasonV2 | None = None,
) -> None:
    assert type(value) is gate_v2.Unsealed960AvailableGateMechanicsRejectionV2
    assert value.disposition is gate_v2.Unsealed960AvailableGateMechanicsDispositionV2.REJECTED
    if reason is not None:
        assert value.reason is reason
    assert value.validation is None
    assert value.available_overall_gate_mechanics_results == ()
    assert value.available_slice_gate_mechanics_results == ()
    assert value.unavailable_gate_mechanics == ()
    assert value.partial_output_published is False
    assert all(getattr(value, name) is False for name in (*TRUE_CLAIMS, *FALSE_CLAIMS))


def _oracle_rows(
    mechanics: scoring_v2.Unsealed960PredictionScoringMechanicsV2,
    manifest: input_v2.FormalUnsealedGateInputManifestV2,
) -> tuple[dict[str, int], dict[tuple[str, str, str], tuple[int, int]]]:
    gate_by_input = {item.input_row_id: item for item in manifest.main_gate_input_rows}
    assert len(gate_by_input) == 720
    overall = {
        "family_exact": 0,
        "binding_exact": 0,
        "scale_set_accuracy": 0,
        "joint_exact": 0,
        "hard_negative_rejection": 0,
        "binding_counterfactual_rejection": 0,
        "scale_counterfactual_rejection": 0,
        "sign_or_invariant_break_rejection": 0,
        "abstention_specificity": 0,
        "nonidentifiable_scale_abstention": 0,
    }
    slice_counts: dict[tuple[str, str, str], list[int]] = {}
    control_metric = {
        Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE: "hard_negative_rejection",
        Phase2BCaseType.BINDING_COUNTERFACTUAL: "binding_counterfactual_rejection",
        Phase2BCaseType.SCALE_COUNTERFACTUAL: "scale_counterfactual_rejection",
        Phase2BCaseType.SIGN_OR_INVARIANT_BREAK: "sign_or_invariant_break_rejection",
    }
    for row in mechanics.main_row_results:
        gate = gate_by_input[row.input_row_id]
        if row.case_type in ANSWERABLE_TYPES:
            overall["family_exact"] += int(row.family_exact is True)
            overall["binding_exact"] += int(row.binding_exact is True)
            overall["scale_set_accuracy"] += int(row.scale_set_exact is True)
            overall["joint_exact"] += int(row.joint_exact is True)
        if row.case_type in CONTROL_TYPES:
            success = row.predicted_decision is PredictionDecisionV2.ABSTAIN
            overall[control_metric[row.case_type]] += int(success)
        if row.case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE:
            overall["abstention_specificity"] += int(
                row.predicted_decision is not PredictionDecisionV2.ABSTAIN
            )
        if row.case_type is Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE:
            overall["nonidentifiable_scale_abstention"] += int(
                row.predicted_decision is PredictionDecisionV2.ABSTAIN
            )
        for scope, slice_id in (
            ("family", gate.canonical_family_id.value),
            ("scale", gate.scale_slice_id.value),
        ):
            for metric, eligible, success in (
                ("answerable_joint_exact", row.case_type in ANSWERABLE_TYPES, row.joint_exact is True),
                ("all_control_rejection", row.case_type in CONTROL_TYPES, row.predicted_decision is PredictionDecisionV2.ABSTAIN),
                ("abstention_specificity", row.case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE, row.predicted_decision is not PredictionDecisionV2.ABSTAIN),
            ):
                if not eligible:
                    continue
                counts = slice_counts.setdefault((metric, scope, slice_id), [0, 0])
                counts[0] += int(success)
                counts[1] += 1
    return overall, {key: (value[0], value[1]) for key, value in slice_counts.items()}


def test_public_surface_signature_manifests_and_private_constructors() -> None:
    assert gate_v2.__all__ == (
        "UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_VERSION",
        "UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_CLAIM_LEVEL",
        "UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_SCHEMA_ID",
        "UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_POLICY_ID",
        "Unsealed960AvailableGateMechanicsDispositionV2",
        "Unsealed960AvailableGateMechanicsReasonV2",
        "Unsealed960AvailableGateMechanicsResultV2",
        "Unsealed960UnavailableGateMechanicsV2",
        "Unsealed960AvailableGateMechanicsV2",
        "Unsealed960AvailableGateMechanicsRejectionV2",
        "evaluate_unsealed_960_available_gate_mechanics_v2",
    )
    signature = inspect.signature(gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2)
    assert tuple(signature.parameters) == (
        "scoring_mechanics",
        "replay_input_contract",
        "gate_input_manifest",
    )
    assert all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in signature.parameters.values())
    assert tuple(item.name for item in fields(gate_v2.Unsealed960AvailableGateMechanicsResultV2)) == AVAILABLE_RESULT_FIELDS
    assert tuple(item.name for item in fields(gate_v2.Unsealed960UnavailableGateMechanicsV2)) == UNAVAILABLE_RESULT_FIELDS
    assert tuple(item.name for item in fields(gate_v2.Unsealed960AvailableGateMechanicsV2)) == SUCCESS_FIELDS
    assert tuple(item.name for item in fields(gate_v2.Unsealed960AvailableGateMechanicsRejectionV2)) == REJECTION_FIELDS
    for value_type in (
        gate_v2.Unsealed960AvailableGateMechanicsResultV2,
        gate_v2.Unsealed960UnavailableGateMechanicsV2,
        gate_v2.Unsealed960AvailableGateMechanicsV2,
        gate_v2.Unsealed960AvailableGateMechanicsRejectionV2,
    ):
        with pytest.raises(TypeError):
            value_type()


def test_exact_non_authoritative_identity_and_enums() -> None:
    assert gate_v2.UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_VERSION == "hegel-machine-phase2b-unsealed-960-available-gate-mechanics/2"
    assert gate_v2.UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_CLAIM_LEVEL == "NON_AUTHORITATIVE_AVAILABLE_GATE_MECHANICS_ONLY"
    assert gate_v2.UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_SCHEMA_ID == (
        "phase2b_unsealed_960_available_gate_mechanics_schema_v2_"
        "401bee5f7df3cb1a5b7b93bb23af5d482c82e178b1b047982d2fda11696cf0e9"
    )
    assert gate_v2.UNSEALED_960_AVAILABLE_GATE_MECHANICS_V2_POLICY_ID == (
        "phase2b_unsealed_960_available_gate_mechanics_policy_v2_"
        "ba0d17ddf4a4716ded51eb0d1e997770db1760dece3dc4505410bda775d756c0"
    )
    assert tuple(item.value for item in gate_v2.Unsealed960AvailableGateMechanicsDispositionV2) == (
        "AVAILABLE_GATE_MECHANICS_COMPLETE_NOT_FORMAL_GATE_EVALUATION",
        "REJECTED",
    )
    assert tuple(item.value for item in gate_v2.Unsealed960AvailableGateMechanicsReasonV2) == (
        "TEN_OVERALL_AND_TWENTY_FOUR_SLICE_MECHANICS_COMPLETE",
        "WRONG_INPUT_TYPE",
        "CROSS_VERSION_INPUT",
        "SCORING_MECHANICS_INVALID",
        "REPLAY_INPUT_CONTRACT_INVALID",
        "GATE_INPUT_MANIFEST_INVALID",
        "CROSS_BINDING_MISMATCH",
        "ROW_JOIN_OR_QUOTA_MISMATCH",
        "INTERNAL_ERROR",
    )


def test_fixture_is_mixed_balanced_and_independent_oracle_is_exact(
    supplied_graphs: tuple[object, object, object],
) -> None:
    mechanics, replay, manifest = supplied_graphs
    assert type(mechanics) is scoring_v2.Unsealed960PredictionScoringMechanicsV2
    assert type(replay) is input_v2.ActualUnsealed960ReplayInputContractV2
    assert type(manifest) is input_v2.FormalUnsealedGateInputManifestV2
    assert len(mechanics.main_row_results) == 720
    assert len({item.input_row_id for item in mechanics.main_row_results}) == 720
    assert len({item.answer_row_id for item in mechanics.main_row_results}) == 720
    assert len({item.prediction_record_id for item in mechanics.main_row_results}) == 720
    assert set(item.predicted_decision for item in mechanics.main_row_results) == set(PredictionDecisionV2)
    assert mechanics.prediction_archive_policy_id == (
        "phase2b_recognizer_prediction_archive_policy_v2_"
        "925a7e62d285ae8ea58b6c2f4ddea5111fa7482ec7957b82476b1341b41b905b"
    )
    for row in mechanics.main_row_results:
        answer_mapping = _answer_row_mapping(row)
        stored_answer_id = answer_mapping.pop("answer_row_id")
        assert stored_answer_id == _domain_id(
            answer_mapping,
            domain=ANSWER_ROW_DOMAIN,
            prefix=ANSWER_ROW_PREFIX,
        )
    expected_answer_id, expected_answer_sha, expected_answer_root = (
        _answer_manifest_identity(mechanics.main_row_results, replay)
    )
    assert (
        mechanics.answer_manifest_id,
        mechanics.answer_manifest_sha256,
        mechanics.main_answer_row_ids_root,
    ) == (expected_answer_id, expected_answer_sha, expected_answer_root)
    assert (
        replay.answer_manifest_id,
        replay.answer_manifest_sha256,
        replay.main_answer_row_ids_root,
    ) == (expected_answer_id, expected_answer_sha, expected_answer_root)
    assert (
        manifest.answer_manifest_id,
        manifest.answer_manifest_sha256,
        manifest.main_answer_row_ids_root,
    ) == (expected_answer_id, expected_answer_sha, expected_answer_root)
    overall, slices = _oracle_rows(mechanics, manifest)
    assert overall == {
        "family_exact": 213,
        "binding_exact": 213,
        "scale_set_accuracy": 213,
        "joint_exact": 183,
        "hard_negative_rejection": 84,
        "binding_counterfactual_rejection": 84,
        "scale_counterfactual_rejection": 84,
        "sign_or_invariant_break_rejection": 84,
        "abstention_specificity": 216,
        "nonidentifiable_scale_abstention": 84,
    }
    assert len(slices) == 24
    assert {total for (metric, scope, _), (_, total) in slices.items() if scope == "family" and metric == "answerable_joint_exact"} == {40}
    assert {total for (metric, scope, _), (_, total) in slices.items() if scope == "family" and metric == "all_control_rejection"} == {64}
    assert {total for (metric, scope, _), (_, total) in slices.items() if scope == "family" and metric == "abstention_specificity"} == {38}
    assert {total for (metric, scope, _), (_, total) in slices.items() if scope == "scale" and metric == "answerable_joint_exact"} == {120}
    assert {total for (metric, scope, _), (_, total) in slices.items() if scope == "scale" and metric == "all_control_rejection"} == {192}
    assert {total for (metric, scope, _), (_, total) in slices.items() if scope == "scale" and metric == "abstention_specificity"} == {114}


def test_upstream_nine_metric_results_explicitly_project_to_available_overall_oracle(
    supplied_graphs: tuple[object, object, object],
) -> None:
    mechanics, _replay, manifest = supplied_graphs
    overall, _ = _oracle_rows(mechanics, manifest)  # type: ignore[arg-type]
    upstream = {
        item.metric_name: item
        for item in mechanics.metric_results  # type: ignore[attr-defined]
    }
    projection = {
        "family_exact": "family_exact_accuracy",
        "binding_exact": "binding_exact_accuracy",
        "scale_set_accuracy": "scale_set_accuracy",
        "joint_exact": "joint_exact_accuracy",
        "abstention_specificity": "abstention_specificity",
        "nonidentifiable_scale_abstention": (
            "nonidentifiability_abstention_accuracy"
        ),
    }
    assert tuple(upstream) == tuple(item.metric_name for item in mechanics.metric_results)  # type: ignore[attr-defined]
    assert len(upstream) == 9
    for available_name, upstream_name in projection.items():
        metric = upstream[upstream_name]
        assert metric.observed_denominator == next(
            denominator
            for name, denominator, _point, _wilson in OVERALL_THRESHOLDS
            if name == available_name
        )
        assert metric.success_count == overall[available_name]


def test_positive_composition_materializes_only_available_mechanics(
    supplied_graphs: tuple[object, object, object],
) -> None:
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(supplied_graphs)
    )
    assert type(result) is gate_v2.Unsealed960AvailableGateMechanicsV2, (
        getattr(result, "reason", None)
    )
    assert result.disposition is (
        gate_v2.Unsealed960AvailableGateMechanicsDispositionV2
        .AVAILABLE_GATE_MECHANICS_COMPLETE_NOT_FORMAL_GATE_EVALUATION
    )
    assert result.reason is (
        gate_v2.Unsealed960AvailableGateMechanicsReasonV2
        .TEN_OVERALL_AND_TWENTY_FOUR_SLICE_MECHANICS_COMPLETE
    )
    assert (
        result.main_row_count,
        result.semantic_conflict_excluded_count,
        result.overall_result_count,
        result.slice_result_count,
        result.unavailable_result_count,
    ) == (720, 240, 10, 24, 2)
    assert len(result.available_overall_gate_mechanics_results) == 10
    assert len(result.available_slice_gate_mechanics_results) == 24
    assert len(result.unavailable_gate_mechanics) == 2
    assert all(getattr(result, name) is True for name in TRUE_CLAIMS)
    assert all(getattr(result, name) is False for name in FALSE_CLAIMS)


def test_exact_ten_overall_integer_wilson_threshold_and_id_oracle(
    supplied_graphs: tuple[object, object, object],
) -> None:
    mechanics, replay, manifest = supplied_graphs
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(supplied_graphs)
    )
    assert type(result) is gate_v2.Unsealed960AvailableGateMechanicsV2
    overall, _ = _oracle_rows(mechanics, manifest)  # type: ignore[arg-type]
    definitions = {
        item.gate_name: item
        for item in replay.available_overall_gate_input_definitions  # type: ignore[attr-defined]
    }
    assert tuple(item.metric_name for item in result.available_overall_gate_mechanics_results) == tuple(
        item[0] for item in OVERALL_THRESHOLDS
    )
    for actual, (name, denominator, point_minimum, wilson_minimum) in zip(
        result.available_overall_gate_mechanics_results,
        OVERALL_THRESHOLDS,
        strict=True,
    ):
        successes = overall[name]
        point = successes / denominator
        lower = _wilson(successes, denominator)
        point_pass = successes * point_minimum[1] >= denominator * point_minimum[0]
        wilson_pass = lower >= wilson_minimum[0] / wilson_minimum[1]
        assert actual.scope == "overall"
        assert actual.slice_id is None
        assert actual.gate_input_definition_id == definitions[name].definition_id
        assert (actual.successes, actual.total, actual.expected_denominator) == (
            successes,
            denominator,
            denominator,
        )
        assert actual.minimum_point_estimate_ratio == _ratio(point_minimum)
        assert actual.minimum_wilson_lcb_ratio == _ratio(wilson_minimum)
        assert actual.point_estimate_ratio == _ratio((successes, denominator))
        assert actual.point_estimate_hex == point.hex()
        assert actual.one_sided_wilson_lcb_hex == lower.hex()
        assert actual.point_threshold_passed is point_pass
        assert actual.wilson_threshold_passed is wilson_pass
        assert actual.available_gate_passed is (point_pass and wilson_pass)
        expected_id = _domain_id(
            tuple(
                (field_name, object.__getattribute__(actual, field_name))
                for field_name in AVAILABLE_RESULT_FIELDS[:-1]
            ),
            domain=b"HEGEL/PHASE2B/AVAILABLE_GATE/RESULT/V2\x00",
            prefix="phase2b_unsealed_960_available_gate_result_v2_",
        )
        assert actual.result_id == expected_id


def test_exact_twenty_four_slice_order_denominators_wilson_and_id_oracle(
    supplied_graphs: tuple[object, object, object],
) -> None:
    mechanics, replay, manifest = supplied_graphs
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(supplied_graphs)
    )
    assert type(result) is gate_v2.Unsealed960AvailableGateMechanicsV2
    _, oracle = _oracle_rows(mechanics, manifest)  # type: ignore[arg-type]
    definitions = {
        (item.scope, item.gate_name): item
        for item in replay.slice_gate_input_definitions  # type: ignore[attr-defined]
    }
    expected_order = tuple(
        (name, "family", family.value, denominator, point, wilson)
        for family in CanonicalFamilyId
        for name, _scope, denominator, point, wilson in SLICE_THRESHOLDS[:3]
    ) + tuple(
        (name, "scale", scale.value, denominator, point, wilson)
        for scale in input_v2.FormalUnsealedScaleSliceIdV2
        for name, _scope, denominator, point, wilson in SLICE_THRESHOLDS[3:]
    )
    assert len(expected_order) == 24
    assert tuple(
        (item.metric_name, item.scope, item.slice_id)
        for item in result.available_slice_gate_mechanics_results
    ) == tuple(item[:3] for item in expected_order)
    for actual, (name, scope, slice_id, denominator, point_minimum, wilson_minimum) in zip(
        result.available_slice_gate_mechanics_results,
        expected_order,
        strict=True,
    ):
        successes, observed = oracle[(name, scope, slice_id)]
        assert observed == denominator
        lower = _wilson(successes, denominator)
        point_pass = successes * point_minimum[1] >= denominator * point_minimum[0]
        wilson_pass = lower >= wilson_minimum[0] / wilson_minimum[1]
        assert actual.gate_input_definition_id == definitions[(scope, name)].definition_id
        assert (actual.successes, actual.total, actual.expected_denominator) == (
            successes,
            denominator,
            denominator,
        )
        assert actual.minimum_point_estimate_ratio == _ratio(point_minimum)
        assert actual.minimum_wilson_lcb_ratio == _ratio(wilson_minimum)
        assert actual.point_estimate_ratio == _ratio((successes, denominator))
        assert actual.point_estimate_hex == (successes / denominator).hex()
        assert actual.one_sided_wilson_lcb_hex == lower.hex()
        assert actual.point_threshold_passed is point_pass
        assert actual.wilson_threshold_passed is wilson_pass
        assert actual.available_gate_passed is (point_pass and wilson_pass)
        assert actual.result_id == _domain_id(
            tuple(
                (field_name, object.__getattribute__(actual, field_name))
                for field_name in AVAILABLE_RESULT_FIELDS[:-1]
            ),
            domain=b"HEGEL/PHASE2B/AVAILABLE_GATE/RESULT/V2\x00",
            prefix="phase2b_unsealed_960_available_gate_result_v2_",
        )


def test_two_missing_gates_are_none_not_zero_and_ids_are_independent(
    supplied_graphs: tuple[object, object, object],
) -> None:
    _mechanics, replay, _manifest = supplied_graphs
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(supplied_graphs)
    )
    assert type(result) is gate_v2.Unsealed960AvailableGateMechanicsV2
    definitions = {
        item.gate_name: item
        for item in replay.unavailable_overall_gate_input_definitions  # type: ignore[attr-defined]
    }
    expected = (
        (
            "fail_closed_rate",
            "1/1",
            None,
            "durable_attempt_event_manifest_not_supplied",
        ),
        (
            "preservation_consistency",
            "97/100",
            "47/50",
            "preservation_pair_result_manifest_not_supplied",
        ),
    )
    assert tuple(item.metric_name for item in result.unavailable_gate_mechanics) == tuple(
        item[0] for item in expected
    )
    for actual, (name, point_minimum, wilson_minimum, missing_reason) in zip(
        result.unavailable_gate_mechanics,
        expected,
        strict=True,
    ):
        assert actual.scope == "overall"
        assert actual.gate_input_definition_id == definitions[name].definition_id
        assert actual.minimum_point_estimate_ratio == point_minimum
        assert actual.minimum_wilson_lcb_ratio == wilson_minimum
        assert actual.missing_input_reason == missing_reason
        for field_name in (
            "expected_denominator",
            "successes",
            "total",
            "point_estimate_ratio",
            "point_estimate_hex",
            "one_sided_wilson_lcb_hex",
            "point_threshold_passed",
            "wilson_threshold_passed",
            "available_gate_passed",
        ):
            assert object.__getattribute__(actual, field_name) is None
        assert actual.unavailable_id == _domain_id(
            tuple(
                (field_name, object.__getattribute__(actual, field_name))
                for field_name in UNAVAILABLE_RESULT_FIELDS[:-1]
            ),
            domain=b"HEGEL/PHASE2B/AVAILABLE_GATE/UNAVAILABLE/V2\x00",
            prefix="phase2b_unsealed_960_unavailable_gate_v2_",
        )


def test_success_identity_is_declared_order_content_addressed(
    supplied_graphs: tuple[object, object, object],
) -> None:
    mechanics, replay, manifest = supplied_graphs
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(supplied_graphs)
    )
    assert type(result) is gate_v2.Unsealed960AvailableGateMechanicsV2
    assert result.scoring_mechanics_result_id == mechanics.result_id  # type: ignore[attr-defined]
    assert result.replay_input_contract_result_id == replay.result_id  # type: ignore[attr-defined]
    assert result.gate_input_manifest_id == manifest.gate_input_manifest_id  # type: ignore[attr-defined]
    assert result.scoring_mechanics_schema_id == scoring_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_SCHEMA_ID
    assert result.scoring_mechanics_policy_id == scoring_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_POLICY_ID
    assert result.scoring_mechanics_version == scoring_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_VERSION
    assert result.scoring_mechanics_claim_level == scoring_v2.UNSEALED_960_PREDICTION_SCORING_MECHANICS_V2_CLAIM_LEVEL
    assert result.replay_input_contract_schema_id == input_v2.ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_SCHEMA_ID
    assert result.replay_input_contract_policy_id == input_v2.ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_POLICY_ID
    assert result.replay_input_contract_version == input_v2.ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_VERSION
    assert result.replay_input_contract_claim_level == input_v2.ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL
    assert result.gate_input_manifest_schema_version == input_v2.FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_VERSION
    assert result.gate_input_manifest_claim_level == input_v2.ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL
    assert result.formal_scoring_contract_schema_id == formal_v2.FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_SCHEMA_ID
    assert result.formal_scoring_contract_policy_id == formal_v2.FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_POLICY_ID
    assert result.formal_scoring_contract_version == formal_v2.FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_VERSION
    assert result.formal_scoring_contract_claim_level == formal_v2.FORMAL_UNSEALED_PREDICTION_SCORING_CONTRACT_V2_CLAIM_LEVEL
    preimage = tuple(
        (field_name, _plain(object.__getattribute__(result, field_name)))
        for field_name in SUCCESS_FIELDS
        if field_name != "result_id"
    )
    assert result.result_id == _domain_id(
        preimage,
        domain=b"HEGEL/PHASE2B/AVAILABLE_GATE/MECHANICS/V2\x00",
        prefix="phase2b_unsealed_960_available_gate_mechanics_v2_",
    )


def test_challenge_prediction_identity_changes_cannot_change_available_results(
    supplied_graphs: tuple[object, object, object],
) -> None:
    mechanics, replay, manifest = supplied_graphs
    baseline = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(supplied_graphs)
    )
    changed_mechanics = _rehash_scoring(
        mechanics,  # type: ignore[arg-type]
        prediction_archive_id=(
            "phase2b_recognizer_prediction_archive_v2_"
            + hashlib.sha256(b"all 240 challenge predictions changed").hexdigest()
        ),
        prediction_archive_sha256=hashlib.sha256(
            b"different supplied archive with unchanged main outcomes"
        ).hexdigest(),
    )
    changed = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        scoring_mechanics=changed_mechanics,
        replay_input_contract=replay,  # type: ignore[arg-type]
        gate_input_manifest=manifest,  # type: ignore[arg-type]
    )
    assert type(baseline) is gate_v2.Unsealed960AvailableGateMechanicsV2
    assert type(changed) is gate_v2.Unsealed960AvailableGateMechanicsV2
    assert changed.result_id != baseline.result_id
    assert changed.available_overall_gate_mechanics_results == baseline.available_overall_gate_mechanics_results
    assert changed.available_slice_gate_mechanics_results == baseline.available_slice_gate_mechanics_results
    assert changed.unavailable_gate_mechanics == baseline.unavailable_gate_mechanics
    assert changed.challenge_in_main_denominator is False
    assert changed.challenge_scoring_performed is False


@pytest.mark.parametrize(
    "case",
    (
        "row_boolean",
        "row_answer_crosslink",
        "metric_aggregate",
        "row_reorder",
        "row_duplicate",
        "authority_claim",
    ),
)
def test_correctly_rehashed_fabricated_upstream_success_graphs_reject_atomically(
    supplied_graphs: tuple[object, object, object],
    case: str,
) -> None:
    mechanics, replay, manifest = supplied_graphs
    rows = list(mechanics.main_row_results)  # type: ignore[attr-defined]
    metrics = list(mechanics.metric_results)  # type: ignore[attr-defined]
    changes: dict[str, object] = {}
    if case == "row_boolean":
        rows[0] = _rehash_row(rows[0], joint_exact=not rows[0].joint_exact)
        changes["main_row_results"] = tuple(rows)
    elif case == "row_answer_crosslink":
        rows[0] = _rehash_row(rows[0], answer_row_id=rows[1].answer_row_id)
        changes["main_row_results"] = tuple(rows)
    elif case == "metric_aggregate":
        target = next(
            index
            for index, item in enumerate(metrics)
            if item.metric_name == "family_exact_accuracy"
        )
        metrics[target] = _rehash_metric(
            metrics[target], success_count=metrics[target].success_count + 1
        )
        changes["metric_results"] = tuple(metrics)
    elif case == "row_reorder":
        rows[0], rows[1] = rows[1], rows[0]
        changes["main_row_results"] = tuple(rows)
    elif case == "row_duplicate":
        rows[-1] = rows[0]
        changes["main_row_results"] = tuple(rows)
    elif case == "authority_claim":
        changes["actual_960_case_run_verified"] = True
    else:
        raise AssertionError(case)
    fabricated = _rehash_scoring(mechanics, **changes)  # type: ignore[arg-type]
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        scoring_mechanics=fabricated,
        replay_input_contract=replay,  # type: ignore[arg-type]
        gate_input_manifest=manifest,  # type: ignore[arg-type]
    )
    _assert_atomic_rejection(result)


def test_forged_answer_row_rehashed_through_all_three_top_graphs_still_rejects(
    supplied_graphs: tuple[object, object, object],
) -> None:
    mechanics, replay, manifest = supplied_graphs
    scoring_rows = list(mechanics.main_row_results)  # type: ignore[attr-defined]
    forged_answer_id = ANSWER_ROW_PREFIX + hashlib.sha256(
        b"forged answer identity with every enclosing graph rehashed"
    ).hexdigest()
    assert forged_answer_id not in {item.answer_row_id for item in scoring_rows}
    target_input_row_id = scoring_rows[0].input_row_id
    scoring_rows[0] = _rehash_row(
        scoring_rows[0],
        answer_row_id=forged_answer_id,
    )
    answer_id, answer_sha, answer_root = _answer_manifest_identity(
        tuple(scoring_rows),
        replay,  # type: ignore[arg-type]
    )
    forged_scoring = _rehash_scoring(
        mechanics,  # type: ignore[arg-type]
        main_row_results=tuple(scoring_rows),
        answer_manifest_id=answer_id,
        answer_manifest_sha256=answer_sha,
        main_answer_row_ids_root=answer_root,
    )

    gate_rows = list(manifest.main_gate_input_rows)  # type: ignore[attr-defined]
    target_index = next(
        index
        for index, item in enumerate(gate_rows)
        if item.input_row_id == target_input_row_id
    )
    gate_rows[target_index] = _rehash_gate_row(
        gate_rows[target_index],
        answer_row_id=forged_answer_id,
    )
    gate_root = _sequence_root(
        tuple(item.gate_input_row_id for item in gate_rows),
        domain=b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_ROW_IDS/V2\x00",
        prefix="phase2b_actual_replay_gate_input_rows_v2_",
    )
    forged_manifest = _rehash_gate_manifest(
        manifest,  # type: ignore[arg-type]
        answer_manifest_id=answer_id,
        answer_manifest_sha256=answer_sha,
        main_answer_row_ids_root=answer_root,
        main_gate_input_rows=tuple(gate_rows),
        main_gate_input_row_ids_root=gate_root,
    )
    forged_replay = _rehash_replay(
        replay,  # type: ignore[arg-type]
        answer_manifest_id=answer_id,
        answer_manifest_sha256=answer_sha,
        main_answer_row_ids_root=answer_root,
        gate_input_manifest_id=forged_manifest.gate_input_manifest_id,
        gate_input_manifest_sha256=forged_manifest.gate_input_manifest_sha256,
        main_gate_input_row_ids_root=gate_root,
    )

    assert forged_scoring.result_id != mechanics.result_id  # type: ignore[attr-defined]
    assert forged_replay.result_id != replay.result_id  # type: ignore[attr-defined]
    assert forged_manifest.gate_input_manifest_id != manifest.gate_input_manifest_id  # type: ignore[attr-defined]
    assert len({item.answer_row_id for item in scoring_rows}) == 720
    assert len({item.gate_input_row_id for item in gate_rows}) == 720
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        scoring_mechanics=forged_scoring,
        replay_input_contract=forged_replay,
        gate_input_manifest=forged_manifest,
    )
    _assert_atomic_rejection(
        result,
        gate_v2.Unsealed960AvailableGateMechanicsReasonV2.CROSS_BINDING_MISMATCH,
    )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        (
            "input_archive_id",
            "phase2b_recognizer_input_archive_v2_" + "a" * 64,
        ),
        ("input_archive_sha256", "b" * 64),
        ("batch_id", "phase2b_trusted_wire_batch_v2_" + "c" * 64),
        (
            "execution_freeze_manifest_id",
            "phase2b_execution_freeze_" + "d" * 64,
        ),
        ("protocol_id", "phase2b_protocol_" + "e" * 64),
        (
            "ordered_archive_input_row_ids_root",
            "phase2b_prediction_input_rows_v2_" + "1" * 64,
        ),
        (
            "partition_union_row_ids_root",
            "phase2b_unsealed_partition_union_rows_v2_" + "2" * 64,
        ),
        (
            "formal_scoring_contract_id",
            "phase2b_formal_unsealed_prediction_scoring_contract_v2_"
            + "3" * 64,
        ),
    ),
)
def test_correctly_rehashed_cross_binding_splices_on_previously_omitted_fields_reject(
    supplied_graphs: tuple[object, object, object],
    field_name: str,
    replacement: str,
) -> None:
    mechanics, replay, manifest = supplied_graphs
    changed = _rehash_scoring(
        mechanics,  # type: ignore[arg-type]
        **{field_name: replacement},
    )
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        scoring_mechanics=changed,
        replay_input_contract=replay,  # type: ignore[arg-type]
        gate_input_manifest=manifest,  # type: ignore[arg-type]
    )
    _assert_atomic_rejection(
        result,
        (
            gate_v2.Unsealed960AvailableGateMechanicsReasonV2.SCORING_MECHANICS_INVALID
            if field_name in {"protocol_id", "formal_scoring_contract_id"}
            else gate_v2.Unsealed960AvailableGateMechanicsReasonV2.CROSS_BINDING_MISMATCH
        ),
    )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        (
            "input_archive_version",
            "hegel-machine-phase2b-trusted-recognizer-input-archive/999",
        ),
        (
            "input_archive_policy_id",
            "phase2b_recognizer_input_archive_policy_v2_" + "4" * 64,
        ),
        (
            "batch_policy_id",
            "phase2b_trusted_wire_batch_v2_policy_" + "5" * 64,
        ),
        ("exact_freeze_id", "phase2b_exact_freeze_" + "6" * 64),
        (
            "gate_input_manifest_schema_id",
            "phase2b_formal_unsealed_gate_input_manifest_schema_v2_"
            + "7" * 64,
        ),
        (
            "gate_input_manifest_policy_id",
            "phase2b_formal_unsealed_gate_input_manifest_policy_v2_"
            + "8" * 64,
        ),
    ),
)
def test_correctly_rehashed_replay_policy_and_freeze_cross_binding_splices_reject(
    supplied_graphs: tuple[object, object, object],
    field_name: str,
    replacement: str,
) -> None:
    mechanics, replay, manifest = supplied_graphs
    changed = _rehash_replay(
        replay,  # type: ignore[arg-type]
        **{field_name: replacement},
    )
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        scoring_mechanics=mechanics,  # type: ignore[arg-type]
        replay_input_contract=changed,
        gate_input_manifest=manifest,  # type: ignore[arg-type]
    )
    _assert_atomic_rejection(
        result,
        gate_v2.Unsealed960AvailableGateMechanicsReasonV2.REPLAY_INPUT_CONTRACT_INVALID,
    )


@pytest.mark.parametrize("case", ("case_type", "decision", "family"))
def test_str_enum_impersonators_with_equal_values_reject_exact_type_whitelist(
    supplied_graphs: tuple[object, object, object],
    case: str,
) -> None:
    mechanics, replay, manifest = supplied_graphs
    rows = list(mechanics.main_row_results)  # type: ignore[attr-defined]
    changes: dict[str, object]
    if case == "case_type":
        changes = {"case_type": _FakeCaseType.UNIQUE_SCALE_ANSWERABLE}
    elif case == "decision":
        changes = {"predicted_decision": _FakeDecision.ANSWER}
    elif case == "family":
        changes = {"predicted_canonical_family_id": _FakeFamily.F01}
    else:
        raise AssertionError(case)
    rows[0] = _rehash_row(rows[0], **changes)
    fabricated = _rehash_scoring(
        mechanics,  # type: ignore[arg-type]
        main_row_results=tuple(rows),
    )
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        scoring_mechanics=fabricated,
        replay_input_contract=replay,  # type: ignore[arg-type]
        gate_input_manifest=manifest,  # type: ignore[arg-type]
    )
    _assert_atomic_rejection(
        result,
        gate_v2.Unsealed960AvailableGateMechanicsReasonV2.SCORING_MECHANICS_INVALID,
    )


def _required_top_scalar_cases() -> tuple[tuple[str, str], ...]:
    exclusions = {
        "scoring_mechanics": {
            "metric_results",
            "main_row_results",
            "gate_results",
            "scale_regret_result",
            "bootstrap_result",
        },
        "replay_input_contract": {
            "required_evidence_inventory",
            "available_overall_gate_input_definitions",
            "unavailable_overall_gate_input_definitions",
            "slice_gate_input_definitions",
            "metric_results",
            "scored_rows",
            "gate_results",
            "scale_regret_result",
            "bootstrap_result",
        },
        "gate_input_manifest": {
            "main_gate_input_rows",
            "required_evidence_inventory",
        },
    }
    types = {
        "scoring_mechanics": scoring_v2.Unsealed960PredictionScoringMechanicsV2,
        "replay_input_contract": input_v2.ActualUnsealed960ReplayInputContractV2,
        "gate_input_manifest": input_v2.FormalUnsealedGateInputManifestV2,
    }
    return tuple(
        (graph_name, item.name)
        for graph_name, value_type in types.items()
        for item in fields(value_type)
        if item.name not in exclusions[graph_name]
    )


@pytest.mark.parametrize(("graph_name", "field_name"), _required_top_scalar_cases())
def test_every_required_top_scalar_rejects_none_atomically(
    supplied_graphs: tuple[object, object, object],
    graph_name: str,
    field_name: str,
) -> None:
    graph_index = {
        "scoring_mechanics": 0,
        "replay_input_contract": 1,
        "gate_input_manifest": 2,
    }[graph_name]
    malformed = _unchecked_copy(supplied_graphs[graph_index], **{field_name: None})
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(supplied_graphs, **{graph_name: malformed})
    )
    _assert_atomic_rejection(result)


@pytest.mark.parametrize(
    ("graph_name", "field_name", "expected_prefix"),
    tuple(
        ("scoring_mechanics", field_name, prefix)
        for field_name, prefix in SCORING_ADDRESS_PREFIXES.items()
    )
    + tuple(
        ("replay_input_contract", field_name, prefix)
        for field_name, prefix in REPLAY_ADDRESS_PREFIXES.items()
    )
    + tuple(
        ("gate_input_manifest", field_name, prefix)
        for field_name, prefix in MANIFEST_ADDRESS_PREFIXES.items()
    ),
)
def test_every_top_content_address_enforces_its_exact_prefix_before_hashing(
    supplied_graphs: tuple[object, object, object],
    monkeypatch: pytest.MonkeyPatch,
    graph_name: str,
    field_name: str,
    expected_prefix: str,
) -> None:
    graph_index = {
        "scoring_mechanics": 0,
        "replay_input_contract": 1,
        "gate_input_manifest": 2,
    }[graph_name]
    original = object.__getattribute__(supplied_graphs[graph_index], field_name)
    assert type(original) is str and original.startswith(expected_prefix)
    malformed = _unchecked_copy(
        supplied_graphs[graph_index],
        **{field_name: "phase2b_wrong_namespace_v2_" + "a" * 64},
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise _PrehashBoundaryReached("wrong namespace reached hashing")

    monkeypatch.setattr(gate_v2, "_stable_id", forbidden)
    monkeypatch.setattr(gate_v2, "stable_hash", forbidden)
    monkeypatch.setattr(gate_v2, "canonical_json", forbidden)
    monkeypatch.setattr(gate_v2.hashlib, "sha256", forbidden)
    monkeypatch.setattr(gate_v2, "_wilson_lower_bound", forbidden)
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(supplied_graphs, **{graph_name: malformed})
    )
    _assert_atomic_rejection(result)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("run_context_id", 7),
        ("run_context_id", True),
        ("structural_receipt_id", 7),
        ("structural_receipt_id", True),
        ("partition_manifest_id", 7),
        ("partition_manifest_id", True),
    ),
)
def test_correctly_rehashed_scoring_only_address_type_drift_rejects(
    supplied_graphs: tuple[object, object, object],
    field_name: str,
    replacement: object,
) -> None:
    mechanics, replay, manifest = supplied_graphs
    malformed = _rehash_scoring(
        mechanics,  # type: ignore[arg-type]
        **{field_name: replacement},
    )
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        scoring_mechanics=malformed,
        replay_input_contract=replay,  # type: ignore[arg-type]
        gate_input_manifest=manifest,  # type: ignore[arg-type]
    )
    _assert_atomic_rejection(
        result,
        gate_v2.Unsealed960AvailableGateMechanicsReasonV2.SCORING_MECHANICS_INVALID,
    )


@pytest.mark.parametrize(
    ("graph_name", "field_name", "replacement"),
    (
        ("replay_input_contract", "gate_input_manifest_sha256", 7),
        ("replay_input_contract", "main_row_count", "720"),
        ("replay_input_contract", "exact_contract_identity_verified", 1),
        ("gate_input_manifest", "input_archive_sha256", 7),
        ("gate_input_manifest", "schema_version", True),
        ("gate_input_manifest", "main_answer_row_ids_root", 7),
    ),
)
def test_replay_and_manifest_representative_top_scalar_type_drift_rejects_prehash(
    supplied_graphs: tuple[object, object, object],
    monkeypatch: pytest.MonkeyPatch,
    graph_name: str,
    field_name: str,
    replacement: object,
) -> None:
    graph_index = {
        "replay_input_contract": 1,
        "gate_input_manifest": 2,
    }[graph_name]
    malformed = _unchecked_copy(
        supplied_graphs[graph_index],
        **{field_name: replacement},
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise _PrehashBoundaryReached("top scalar type drift reached hashing")

    monkeypatch.setattr(gate_v2, "_stable_id", forbidden)
    monkeypatch.setattr(gate_v2, "stable_hash", forbidden)
    monkeypatch.setattr(gate_v2, "canonical_json", forbidden)
    monkeypatch.setattr(gate_v2.hashlib, "sha256", forbidden)
    monkeypatch.setattr(gate_v2, "_wilson_lower_bound", forbidden)
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(supplied_graphs, **{graph_name: malformed})
    )
    _assert_atomic_rejection(result)


@pytest.mark.parametrize(
    ("graph_name", "field_name"),
    (
        ("scoring_mechanics", "version"),
        ("replay_input_contract", "version"),
        ("gate_input_manifest", "schema_version"),
    ),
)
def test_required_scalar_exact_type_preflight_never_invokes_hostile_equality(
    supplied_graphs: tuple[object, object, object],
    monkeypatch: pytest.MonkeyPatch,
    graph_name: str,
    field_name: str,
) -> None:
    graph_index = {
        "scoring_mechanics": 0,
        "replay_input_contract": 1,
        "gate_input_manifest": 2,
    }[graph_name]
    hostile = _HostileEquality()
    malformed = _unchecked_copy(
        supplied_graphs[graph_index],
        **{field_name: hostile},
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise _PrehashBoundaryReached("hostile scalar reached hash or Wilson")

    monkeypatch.setattr(gate_v2, "_stable_id", forbidden)
    monkeypatch.setattr(gate_v2, "stable_hash", forbidden)
    monkeypatch.setattr(gate_v2, "canonical_json", forbidden)
    monkeypatch.setattr(gate_v2.hashlib, "sha256", forbidden)
    monkeypatch.setattr(gate_v2, "_wilson_lower_bound", forbidden)
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(supplied_graphs, **{graph_name: malformed})
    )
    _assert_atomic_rejection(result)
    assert hostile.comparison_attempts == 0
    assert object.__getattribute__(malformed, field_name) is hostile


def _graph_with_nested_cap_value(
    graphs: tuple[object, object, object],
    *,
    case: str,
    over_limit: bool,
) -> tuple[object, object, object]:
    mechanics, replay, manifest = graphs
    rows = list(mechanics.main_row_results)  # type: ignore[attr-defined]
    row = rows[-1]
    if case == "utf8_text":
        outcomes = list(row.metric_outcomes)
        value = "é" * 2_048 + ("a" if over_limit else "")
        assert len(value.encode("utf-8")) == (4_097 if over_limit else 4_096)
        outcomes[-1] = _unchecked_copy(outcomes[-1], metric_name=value)
        row = _unchecked_copy(row, metric_outcomes=tuple(outcomes))
    elif case == "binding_tuple":
        count = 65 if over_limit else 64
        value = tuple(
            RoleBinding(
                role_id=f"00000000-0000-4000-8000-{1_000_000 + index:012x}",
                entity_id=f"00000000-0000-4000-8000-{2_000_000 + index:012x}",
            )
            for index in range(count)
        )
        row = _unchecked_copy(row, predicted_binding=value)
    elif case == "scale_tuple":
        count = 4_097 if over_limit else 4_096
        value = tuple(
            f"00000000-0000-4000-8000-{3_000_000 + index:012x}"
            for index in range(count)
        )
        row = _unchecked_copy(row, predicted_admissible_scale_ids=value)
    else:
        raise AssertionError(case)
    rows[-1] = row
    mechanics = _unchecked_copy(mechanics, main_row_results=tuple(rows))
    return mechanics, replay, manifest


@pytest.mark.parametrize("case", ("utf8_text", "binding_tuple", "scale_tuple"))
def test_nested_cap_exact_boundary_is_accepted_by_its_closed_parser(
    supplied_graphs: tuple[object, object, object],
    case: str,
) -> None:
    boundary_graphs = _graph_with_nested_cap_value(
        supplied_graphs,
        case=case,
        over_limit=False,
    )
    row = boundary_graphs[0].main_row_results[-1]  # type: ignore[attr-defined]
    if case == "utf8_text":
        value = row.metric_outcomes[-1].metric_name
        assert len(value.encode("utf-8")) == 4_096
        assert gate_v2._text(value, "boundary text") == value
    elif case == "binding_tuple":
        closed = gate_v2._binding_snapshot(row.predicted_binding, "boundary binding")
        assert len(closed) == 64
    elif case == "scale_tuple":
        closed = gate_v2._row_snapshot(row)
        assert len(closed["predicted_admissible_scale_ids"]) == 4_096
    else:
        raise AssertionError(case)


@pytest.mark.parametrize("case", ("utf8_text", "binding_tuple", "scale_tuple"))
def test_nested_cap_boundary_plus_one_rejects_before_hashing(
    supplied_graphs: tuple[object, object, object],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    malformed = _graph_with_nested_cap_value(
        supplied_graphs,
        case=case,
        over_limit=True,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise _PrehashBoundaryReached(f"{case} over cap reached hashing")

    monkeypatch.setattr(gate_v2, "_stable_id", forbidden)
    monkeypatch.setattr(gate_v2, "stable_hash", forbidden)
    monkeypatch.setattr(gate_v2, "canonical_json", forbidden)
    monkeypatch.setattr(gate_v2.hashlib, "sha256", forbidden)
    monkeypatch.setattr(gate_v2, "_wilson_lower_bound", forbidden)
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(malformed)
    )
    _assert_atomic_rejection(
        result,
        gate_v2.Unsealed960AvailableGateMechanicsReasonV2.SCORING_MECHANICS_INVALID,
    )


@pytest.mark.parametrize(
    "bad_uuid",
    (
        "not-a-uuid",
        "00000000-0000-4000-8000-00000000000A",
        "00000000-0000-1000-8000-000000000000",
        "00000000000040008000000000000000",
    ),
)
def test_malformed_or_noncanonical_uuid_rejects_before_hashing(
    supplied_graphs: tuple[object, object, object],
    monkeypatch: pytest.MonkeyPatch,
    bad_uuid: str,
) -> None:
    mechanics, replay, manifest = supplied_graphs
    rows = list(mechanics.main_row_results)  # type: ignore[attr-defined]
    malformed_binding = _unchecked_copy(
        RoleBinding(
            role_id="00000000-0000-4000-8000-000000000000",
            entity_id="00000000-0000-4000-8000-000000000001",
        ),
        role_id=bad_uuid,
    )
    rows[-1] = _unchecked_copy(
        rows[-1],
        predicted_binding=(malformed_binding,),
    )
    malformed_mechanics = _unchecked_copy(
        mechanics,
        main_row_results=tuple(rows),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise _PrehashBoundaryReached("invalid UUID reached hashing")

    monkeypatch.setattr(gate_v2, "_stable_id", forbidden)
    monkeypatch.setattr(gate_v2, "stable_hash", forbidden)
    monkeypatch.setattr(gate_v2, "canonical_json", forbidden)
    monkeypatch.setattr(gate_v2.hashlib, "sha256", forbidden)
    monkeypatch.setattr(gate_v2, "_wilson_lower_bound", forbidden)
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        scoring_mechanics=malformed_mechanics,
        replay_input_contract=replay,  # type: ignore[arg-type]
        gate_input_manifest=manifest,  # type: ignore[arg-type]
    )
    _assert_atomic_rejection(
        result,
        gate_v2.Unsealed960AvailableGateMechanicsReasonV2.SCORING_MECHANICS_INVALID,
    )


def _malformed_graphs(
    graphs: tuple[object, object, object], case: str
) -> tuple[object, object, object]:
    mechanics, replay, manifest = graphs
    if case == "last_metric_outcome":
        rows = list(mechanics.main_row_results)  # type: ignore[attr-defined]
        outcomes = list(rows[-1].metric_outcomes)
        outcomes[-1] = _unchecked_copy(outcomes[-1], metric_name=object())
        rows[-1] = _unchecked_copy(rows[-1], metric_outcomes=tuple(outcomes))
        mechanics = _unchecked_copy(mechanics, main_row_results=tuple(rows))
    elif case == "last_required_evidence":
        inventory = list(replay.required_evidence_inventory)  # type: ignore[attr-defined]
        inventory[-1] = _unchecked_copy(inventory[-1], purpose=object())
        replay = _unchecked_copy(replay, required_evidence_inventory=tuple(inventory))
    elif case == "last_gate_row":
        rows = list(manifest.main_gate_input_rows)  # type: ignore[attr-defined]
        rows[-1] = _unchecked_copy(rows[-1], latent_base_case_id=object())
        manifest = _unchecked_copy(manifest, main_gate_input_rows=tuple(rows))
    else:
        raise AssertionError(case)
    return mechanics, replay, manifest


@pytest.mark.parametrize(
    ("case", "reason"),
    (
        (
            "last_metric_outcome",
            gate_v2.Unsealed960AvailableGateMechanicsReasonV2.SCORING_MECHANICS_INVALID,
        ),
        (
            "last_required_evidence",
            gate_v2.Unsealed960AvailableGateMechanicsReasonV2.REPLAY_INPUT_CONTRACT_INVALID,
        ),
        (
            "last_gate_row",
            gate_v2.Unsealed960AvailableGateMechanicsReasonV2.GATE_INPUT_MANIFEST_INVALID,
        ),
    ),
)
def test_global_nested_preflight_precedes_every_hash_and_wilson_operation(
    supplied_graphs: tuple[object, object, object],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    reason: gate_v2.Unsealed960AvailableGateMechanicsReasonV2,
) -> None:
    malformed = _malformed_graphs(supplied_graphs, case)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise _PrehashBoundaryReached("malformed graph reached hash or Wilson")

    monkeypatch.setattr(gate_v2, "_stable_id", forbidden)
    monkeypatch.setattr(gate_v2, "stable_hash", forbidden)
    monkeypatch.setattr(gate_v2, "canonical_json", forbidden)
    monkeypatch.setattr(gate_v2.hashlib, "sha256", forbidden)
    monkeypatch.setattr(gate_v2, "_wilson_lower_bound", forbidden)
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(malformed)
    )
    _assert_atomic_rejection(result, reason)


def test_valid_graph_reaches_post_preflight_hash_boundary(
    supplied_graphs: tuple[object, object, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reached(*args: object, **kwargs: object) -> object:
        raise _PrehashBoundaryReached("valid graph completed global preflight")

    monkeypatch.setattr(gate_v2, "stable_hash", reached)
    with pytest.raises(_PrehashBoundaryReached):
        gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
            **_evaluate_kwargs(supplied_graphs)
        )


def test_closed_snapshot_is_immune_to_hash_time_caller_mutation(
    supplied_graphs: tuple[object, object, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mechanics, replay, manifest = supplied_graphs
    caller_mechanics = _unchecked_copy(mechanics)
    caller_replay = _unchecked_copy(replay)
    caller_manifest = _unchecked_copy(manifest)
    expected_scoring_id = caller_mechanics.result_id  # type: ignore[attr-defined]
    expected_replay_id = caller_replay.result_id  # type: ignore[attr-defined]
    expected_manifest_id = caller_manifest.gate_input_manifest_id  # type: ignore[attr-defined]
    original = gate_v2.stable_hash
    mutations = 0

    def mutate_callers_at_first_hash(*args: object, **kwargs: object) -> object:
        nonlocal mutations
        if mutations == 0:
            mutations += 1
            object.__setattr__(caller_mechanics, "main_row_results", ())
            object.__setattr__(caller_replay, "required_evidence_inventory", ())
            object.__setattr__(caller_manifest, "main_gate_input_rows", ())
        return original(*args, **kwargs)

    monkeypatch.setattr(gate_v2, "stable_hash", mutate_callers_at_first_hash)
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        scoring_mechanics=caller_mechanics,  # type: ignore[arg-type]
        replay_input_contract=caller_replay,  # type: ignore[arg-type]
        gate_input_manifest=caller_manifest,  # type: ignore[arg-type]
    )
    assert mutations == 1
    assert type(result) is gate_v2.Unsealed960AvailableGateMechanicsV2
    assert result.scoring_mechanics_result_id == expected_scoring_id
    assert result.replay_input_contract_result_id == expected_replay_id
    assert result.gate_input_manifest_id == expected_manifest_id
    assert caller_mechanics.main_row_results == ()  # type: ignore[attr-defined]
    assert caller_replay.required_evidence_inventory == ()  # type: ignore[attr-defined]
    assert caller_manifest.main_gate_input_rows == ()  # type: ignore[attr-defined]


def test_success_outputs_are_deep_fresh_across_calls(
    supplied_graphs: tuple[object, object, object],
) -> None:
    first = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(supplied_graphs)
    )
    second = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(supplied_graphs)
    )
    assert type(first) is gate_v2.Unsealed960AvailableGateMechanicsV2
    assert type(second) is gate_v2.Unsealed960AvailableGateMechanicsV2
    assert first == second
    assert first is not second
    for left_group, right_group in (
        (
            first.available_overall_gate_mechanics_results,
            second.available_overall_gate_mechanics_results,
        ),
        (
            first.available_slice_gate_mechanics_results,
            second.available_slice_gate_mechanics_results,
        ),
        (first.unavailable_gate_mechanics, second.unavailable_gate_mechanics),
    ):
        assert left_group is not right_group
        assert all(left is not right for left, right in zip(left_group, right_group, strict=True))
    object.__setattr__(first.available_overall_gate_mechanics_results[0], "metric_name", "caller pollution")
    third = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(supplied_graphs)
    )
    assert type(third) is gate_v2.Unsealed960AvailableGateMechanicsV2
    assert third.available_overall_gate_mechanics_results[0].metric_name == "family_exact"


def test_internal_exception_is_atomic_all_false(
    supplied_graphs: tuple[object, object, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic internal fault")

    monkeypatch.setattr(gate_v2, "_calculate_results", explode)
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        **_evaluate_kwargs(supplied_graphs)
    )
    _assert_atomic_rejection(
        result,
        gate_v2.Unsealed960AvailableGateMechanicsReasonV2.INTERNAL_ERROR,
    )


def test_cross_version_scoring_graph_uses_explicit_cross_version_rejection(
    supplied_graphs: tuple[object, object, object],
) -> None:
    mechanics, replay, manifest = supplied_graphs
    changed = _rehash_scoring(
        mechanics,  # type: ignore[arg-type]
        version="hegel-machine-phase2b-unsealed-960-prediction-scoring-mechanics/999",
    )
    result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        scoring_mechanics=changed,
        replay_input_contract=replay,  # type: ignore[arg-type]
        gate_input_manifest=manifest,  # type: ignore[arg-type]
    )
    _assert_atomic_rejection(
        result,
        gate_v2.Unsealed960AvailableGateMechanicsReasonV2.CROSS_VERSION_INPUT,
    )


def test_source_ast_forbids_upstream_execution_validation_and_operational_calls() -> None:
    source = Path(gate_v2.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        item.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for item in node.names
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden_import_parts = {
        "os",
        "pathlib",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "random",
        "secrets",
        "tempfile",
        "phase2b_runner",
    }
    assert not {
        name
        for name in imported
        if any(part in name.split(".") for part in forbidden_import_parts)
    }
    forbidden_calls = {
        "score_unsealed_960_prediction_scoring_mechanics_v2",
        "validate_actual_unsealed_960_replay_input_contract_v2",
        "build_formal_unsealed_gate_input_manifest_v2",
        "decode_public_recognizer_prediction_archive_v2",
        "decode_public_recognizer_input_archive_v2",
        "frozen_phase2b_protocol",
        "frozen_formal_unsealed_prediction_scoring_contract_v2",
        "evaluate_binary_gate",
        "one_sided_wilson_lower_bound",
        "run_recognizer",
        "open",
        "read_bytes",
        "write_bytes",
    }
    assert calls.isdisjoint(forbidden_calls)


@pytest.mark.parametrize("wrong", [None, object(), (), {}, 0, False, "graph"])
def test_wrong_top_level_type_rejects_atomically(wrong: object, supplied_graphs: tuple[object, object, object]) -> None:
    for name in ("scoring_mechanics", "replay_input_contract", "gate_input_manifest"):
        result = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
            **_evaluate_kwargs(supplied_graphs, **{name: wrong})
        )
        _assert_atomic_rejection(result, gate_v2.Unsealed960AvailableGateMechanicsReasonV2.WRONG_INPUT_TYPE)
