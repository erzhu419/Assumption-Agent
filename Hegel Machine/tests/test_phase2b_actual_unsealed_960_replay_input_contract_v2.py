"""Synthetic tests for the actual-unsealed-960 replay-input contract V2.

The positive fixture is locally fabricated contract data.  It is not an
authenticated evaluator opening, proof of pre-reveal timing or custody, an
actual recognizer run, scoring, gate evaluation, effect evidence, or C1 exit
evidence.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import inspect
from pathlib import Path

import pytest

import hegel_machine.phase2b_actual_unsealed_960_replay_input_contract_v2 as replay_v2
import hegel_machine.phase2b_formal_unsealed_prediction_scoring_contract_v2 as scoring_v2
from hegel_machine.hashing import canonical_json
from hegel_machine.phase2b_freeze_v1 import CanonicalFamilyId, frozen_phase2b_exact_freeze
from hegel_machine.phase2b_protocol import (
    MarginStratum,
    Phase2BCaseType,
    frozen_phase2b_protocol,
    salted_answer_commitment_sha256,
)
from hegel_machine.phase2b_recognizer_input_archive_v2 import (
    RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
    TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
)
from hegel_machine.phase2b_recognizer_prediction_v2 import PredictionDecisionV2
from hegel_machine.phase2b_trusted_wire_batch_v2 import TRUSTED_WIRE_BATCH_V2_POLICY_ID
from hegel_machine.phase2b_wire import RoleBinding


EXPECTED_PUBLIC_SURFACE = [
    "ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_VERSION",
    "ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL",
    "ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_SCHEMA_ID",
    "ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_POLICY_ID",
    "FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_VERSION",
    "FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_ID",
    "FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_POLICY_ID",
    "FormalUnsealedScaleSliceIdV2",
    "ActualUnsealed960ReplayInputDispositionV2",
    "ActualUnsealed960ReplayInputReasonV2",
    "FormalUnsealedGateInputRowV2",
    "ActualReplayGateInputDefinitionV2",
    "ActualReplayRequiredEvidenceV2",
    "FormalUnsealedGateInputManifestV2",
    "ActualUnsealed960ReplayInputContractV2",
    "ActualUnsealed960ReplayInputContractRejectionV2",
    "build_formal_unsealed_gate_input_manifest_v2",
    "salted_gate_input_commitment_sha256_v2",
    "validate_actual_unsealed_960_replay_input_contract_v2",
]
GATE_ROW_FIELDS = (
    "input_row_id",
    "answer_row_id",
    "case_type",
    "margin_stratum",
    "canonical_family_id",
    "scale_slice_id",
    "latent_base_case_id",
    "gate_input_row_id",
)
UPSTREAM_ANSWER_ROW_FIELDS = (
    "input_row_id",
    "case_type",
    "expected_decision",
    "canonical_family_id",
    "binding",
    "admissible_scale_ids",
    "answer_row_id",
)
UPSTREAM_ANSWER_MANIFEST_FIELDS = (
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
GATE_DEFINITION_FIELDS = (
    "gate_name",
    "scope",
    "expected_denominator",
    "success_rule",
    "input_available",
    "missing_input_reason",
    "definition_id",
)
REQUIRED_EVIDENCE_FIELDS = (
    "evidence_name",
    "purpose",
    "supplied_by_this_contract",
    "verifier_implemented",
    "requirement_id",
)
MANIFEST_FIELDS = (
    "schema_version",
    "schema_id",
    "policy_id",
    "claim_level",
    "exact_freeze_id",
    "phase2b_protocol_id",
    "formal_scoring_contract_id",
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
    "answer_manifest_id",
    "answer_manifest_sha256",
    "main_answer_row_ids_root",
    "main_gate_input_rows",
    "main_gate_input_row_ids_root",
    "required_evidence_inventory",
    "gate_input_manifest_sha256",
    "gate_input_manifest_id",
)
TRUE_RESULT_CLAIMS = (
    "exact_contract_identity_verified",
    "answer_gate_manifest_cross_binding_verified",
    "supplied_gate_input_commitment_opening_verified",
    "exact_main_gate_row_coverage_verified",
    "exact_family_scale_cell_quota_verified",
    "exact_case_type_per_cell_quota_verified",
    "exact_margin_per_cell_quota_verified",
    "exact_nonunique_margin_case_composition_verified",
    "supplied_family_slice_labels_complete",
    "supplied_scale_slice_labels_complete",
    "unique_latent_base_case_ids_verified",
    "downstream_prediction_identifier_fields_absent_from_schema_verified",
    "semantic_conflict_root_bound_and_exclusion_contract_frozen",
    "control_gate_input_semantics_frozen",
    "slice_gate_input_semantics_frozen",
    "required_unsupplied_evidence_inventory_frozen",
)
FALSE_RESULT_CLAIMS = (
    "challenge_in_main_denominator",
    "margin_stratum_authority_verified",
    "family_slice_label_authority_verified",
    "scale_slice_semantics_authority_verified",
    "latent_case_independence_verified",
    "one_shot_policy_enforced",
    "durable_attempt_ledger_verified",
    "raw_input_archive_replayed",
    "raw_prediction_archive_replayed",
    "prediction_commit_before_reveal_verified",
    "wilson_bounds_evaluated",
    "preservation_evaluated",
    "challenge_descriptor_rows_implemented",
    "challenge_scoring_performed",
    "fail_closed_gate_inputs_contract_complete",
    "preservation_gate_inputs_contract_complete",
    "scale_regret_inputs_contract_complete",
    "bootstrap_inputs_contract_complete",
    "answer_manifest_authority_verified",
    "gate_input_manifest_authority_verified",
    "answer_commitment_authority_verified",
    "gate_input_commitment_authority_verified",
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
    "formal_gate_evaluation_performed",
    "metric_results_materialized",
    "scored_rows_materialized",
    "overall_gate_results_materialized",
    "slice_gate_results_materialized",
    "scale_regret_evaluated",
    "bootstrap_evaluated",
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
    "gate_input_manifest_id",
    "gate_input_manifest_sha256",
    "salted_gate_input_commitment_sha256",
    "gate_input_manifest_schema_id",
    "gate_input_manifest_policy_id",
    "answer_manifest_id",
    "answer_manifest_sha256",
    "execution_freeze_manifest_id",
    "input_archive_id",
    "input_archive_sha256",
    "input_archive_version",
    "input_archive_policy_id",
    "batch_id",
    "batch_policy_id",
    "exact_freeze_id",
    "protocol_id",
    "formal_scoring_contract_id",
    "ordered_archive_input_row_ids_root",
    "main_row_ids_root",
    "semantic_conflict_row_ids_root",
    "partition_union_row_ids_root",
    "main_answer_row_ids_root",
    "main_gate_input_row_ids_root",
    "main_row_count",
    "semantic_conflict_expected_row_count",
    "total_expected_prediction_count",
    "unique_latent_base_case_id_count",
    "family_scale_cell_count",
    *TRUE_RESULT_CLAIMS,
    *FALSE_RESULT_CLAIMS,
    "required_evidence_inventory",
    "available_overall_gate_input_definitions",
    "unavailable_overall_gate_input_definitions",
    "slice_gate_input_definitions",
    "metric_results",
    "scored_rows",
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
    "validation",
    "required_evidence_inventory",
    "available_overall_gate_input_definitions",
    "unavailable_overall_gate_input_definitions",
    "slice_gate_input_definitions",
    "metric_results",
    "scored_rows",
    "gate_results",
    "scale_regret_result",
    "bootstrap_result",
    "partial_output_published",
    *TRUE_RESULT_CLAIMS,
    *FALSE_RESULT_CLAIMS,
)


EXPECTED_CASE_QUOTA_PER_CELL = {
    Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE: 19,
    Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE: 1,
    Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE: 8,
    Phase2BCaseType.BINDING_COUNTERFACTUAL: 8,
    Phase2BCaseType.SCALE_COUNTERFACTUAL: 8,
    Phase2BCaseType.SIGN_OR_INVARIANT_BREAK: 8,
    Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE: 8,
}
EXPECTED_MARGIN_QUOTA_PER_CELL = {
    MarginStratum.CLEAR_INTERIOR: 21,
    MarginStratum.MODERATE: 18,
    MarginStratum.NEAR_BOUNDARY_IDENTIFIABLE: 12,
    MarginStratum.NONUNIQUE_OR_INSUFFICIENT: 9,
}
EXPECTED_OVERALL_AVAILABLE = (
    "family_exact",
    "binding_exact",
    "scale_set_accuracy",
    "joint_exact",
    "hard_negative_rejection",
    "binding_counterfactual_rejection",
    "scale_counterfactual_rejection",
    "sign_or_invariant_break_rejection",
    "abstention_specificity",
    "nonidentifiable_scale_abstention",
)
EXPECTED_OVERALL_UNAVAILABLE = (
    "fail_closed_rate",
    "preservation_consistency",
)
EXPECTED_SLICE_DENOMINATORS = (
    ("answerable_joint_exact", "family", 40),
    ("all_control_rejection", "family", 64),
    ("abstention_specificity", "family", 38),
    ("answerable_joint_exact", "scale", 120),
    ("all_control_rejection", "scale", 192),
    ("abstention_specificity", "scale", 114),
)
EXPECTED_FORBIDDEN_DOWNSTREAM_FIELDS = frozenset(
    {
        "prediction_archive_id",
        "prediction_archive_sha256",
        "prediction_record_id",
        "prediction_content_id",
        "run_context_id",
        "partition_manifest_id",
        "structural_receipt_id",
        "structural_evaluation_id",
        "runtime_receipt_id",
        "score_id",
        "metric_result_id",
        "scored_row_id",
        "gate_result_id",
        "effect_id",
        "c1_result_id",
        "timestamp",
        "run_started_at",
        "predictions_committed_at",
        "answer_revealed_at",
        "attempt_status",
        "attempt_index",
    }
)
FORBIDDEN_IMPORT_SUFFIXES = frozenset(
    {
        "os",
        "io",
        "tempfile",
        "shutil",
        "asyncio",
        "multiprocessing",
        "concurrent",
        "concurrent.futures",
        "phase2b_runner",
        "phase2b_recognizer_prediction_archive_v2",
        "phase2b_unsealed_prediction_evaluator_v2",
        "phase2b_unsealed_960_prediction_scoring_mechanics_v2",
        "phase2b_strict_recognizer_cli_v2",
        "bootstrap",
        "subprocess",
        "pathlib",
        "socket",
        "requests",
        "urllib",
        "urllib.request",
        "random",
        "secrets",
        "numpy",
        "scipy",
        "statistics",
    }
)
FORBIDDEN_CALL_TERMINALS = frozenset(
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
        "open",
        "read_bytes",
        "write_bytes",
        "Popen",
        "record_generated_holdout",
        "commit_predictions",
        "consume",
        "invalidate",
        "evaluate_preservation",
        "generate_preservation",
    }
)
ANSWERABLE_CASE_TYPES = frozenset(
    {
        Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE,
        Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE,
    }
)


class _ForbiddenBoundaryReached(BaseException):
    """Sentinel proving malformed data reached hashing or finalization."""


class _HostileText(str):
    def encode(self, *args: object, **kwargs: object) -> bytes:
        raise _ForbiddenBoundaryReached("hostile text encoded")


class _HostileTuple(tuple):
    def __iter__(self):  # type: ignore[no-untyped-def]
        raise _ForbiddenBoundaryReached("hostile tuple iterated")


def _hex_id(prefix: str, index: int) -> str:
    return prefix + hashlib.sha256(f"synthetic-v2-{index}".encode("ascii")).hexdigest()


def _uuid4(index: int) -> str:
    raw = f"{index:032x}"[-32:]
    return f"{raw[:8]}-{raw[8:12]}-4{raw[13:16]}-a{raw[17:20]}-{raw[20:]}"


def _row_sequence_root(
    values: tuple[str, ...],
    *,
    domain: bytes,
    prefix: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(len(values).to_bytes(4, "big"))
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return prefix + digest.hexdigest()


def _main_root(values: tuple[str, ...]) -> str:
    return _row_sequence_root(
        values,
        domain=b"HEGEL/PHASE2B/UNSEALED/MAIN_ROWS/V2\x00",
        prefix="phase2b_unsealed_main_rows_v2_",
    )


def _challenge_root(values: tuple[str, ...]) -> str:
    return _row_sequence_root(
        values,
        domain=b"HEGEL/PHASE2B/UNSEALED/SEMANTIC_CONFLICT_ROWS/V2\x00",
        prefix="phase2b_unsealed_semantic_conflict_rows_v2_",
    )


def _union_root(values: tuple[str, ...]) -> str:
    return _row_sequence_root(
        values,
        domain=b"HEGEL/PHASE2B/UNSEALED/PARTITION_UNION_ROWS/V2\x00",
        prefix="phase2b_unsealed_partition_union_rows_v2_",
    )


def _ordered_root(values: tuple[str, ...]) -> str:
    return _row_sequence_root(
        values,
        domain=b"HEGEL/PHASE2B/PREDICTION_INPUT_ROWS/V2\x00",
        prefix="phase2b_prediction_input_rows_v2_",
    )


def _synthetic_answer_row(
    *,
    index: int,
    input_row_id: str,
    case_type: Phase2BCaseType,
    family_id: CanonicalFamilyId,
) -> scoring_v2.FormalUnsealedAnswerRowV2:
    if case_type is Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE:
        decision = PredictionDecisionV2.ANSWER
        answer_family: CanonicalFamilyId | None = family_id
        binding = (
            RoleBinding(
                role_id=_uuid4(10_000 + index),
                entity_id=_uuid4(20_000 + index),
            ),
        )
        scales = (_uuid4(30_000 + index),)
    elif case_type is Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE:
        decision = PredictionDecisionV2.ANSWER_SET
        answer_family = family_id
        binding = (
            RoleBinding(
                role_id=_uuid4(10_000 + index),
                entity_id=_uuid4(20_000 + index),
            ),
        )
        scales = tuple(
            sorted((_uuid4(40_000 + 2 * index), _uuid4(40_001 + 2 * index)))
        )
    else:
        decision = PredictionDecisionV2.ABSTAIN
        answer_family = None
        binding = ()
        scales = ()
    preimage = {
        "input_row_id": input_row_id,
        "case_type": case_type.value,
        "expected_decision": decision.value,
        "canonical_family_id": (
            None if answer_family is None else answer_family.value
        ),
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
        canonical_family_id=answer_family,
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


def _independent_gate_row_mapping(
    row: replay_v2.FormalUnsealedGateInputRowV2,
    *,
    include_row_id: bool,
) -> dict[str, object]:
    mapping: dict[str, object] = {
        "input_row_id": row.input_row_id,
        "answer_row_id": row.answer_row_id,
        "case_type": row.case_type.value,
        "margin_stratum": row.margin_stratum.value,
        "canonical_family_id": row.canonical_family_id.value,
        "scale_slice_id": row.scale_slice_id.value,
        "latent_base_case_id": row.latent_base_case_id,
    }
    if include_row_id:
        mapping["gate_input_row_id"] = row.gate_input_row_id
    return mapping


def _independent_manifest_preimage(
    manifest: replay_v2.FormalUnsealedGateInputManifestV2,
) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "schema_id": manifest.schema_id,
        "policy_id": manifest.policy_id,
        "claim_level": manifest.claim_level,
        "exact_freeze_id": manifest.exact_freeze_id,
        "phase2b_protocol_id": manifest.phase2b_protocol_id,
        "formal_scoring_contract_id": manifest.formal_scoring_contract_id,
        "execution_freeze_manifest_id": manifest.execution_freeze_manifest_id,
        "input_archive_id": manifest.input_archive_id,
        "input_archive_sha256": manifest.input_archive_sha256,
        "input_archive_version": manifest.input_archive_version,
        "input_archive_policy_id": manifest.input_archive_policy_id,
        "batch_id": manifest.batch_id,
        "batch_policy_id": manifest.batch_policy_id,
        "ordered_archive_input_row_ids_root": (
            manifest.ordered_archive_input_row_ids_root
        ),
        "main_row_ids_root": manifest.main_row_ids_root,
        "semantic_conflict_row_ids_root": manifest.semantic_conflict_row_ids_root,
        "partition_union_row_ids_root": manifest.partition_union_row_ids_root,
        "answer_manifest_id": manifest.answer_manifest_id,
        "answer_manifest_sha256": manifest.answer_manifest_sha256,
        "main_answer_row_ids_root": manifest.main_answer_row_ids_root,
        "main_gate_input_rows": [
            _independent_gate_row_mapping(row, include_row_id=True)
            for row in manifest.main_gate_input_rows
        ],
        "main_gate_input_row_ids_root": manifest.main_gate_input_row_ids_root,
        "required_evidence_inventory": [
            {
                name: getattr(item, name)
                for name in REQUIRED_EVIDENCE_FIELDS
            }
            for item in manifest.required_evidence_inventory
        ],
    }


def _independent_result_preimage(
    result: replay_v2.ActualUnsealed960ReplayInputContractV2,
) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for item in fields(type(result)):
        if item.name == "result_id":
            continue
        raw = getattr(result, item.name)
        if isinstance(raw, Enum):
            mapping[item.name] = raw.value
        elif type(raw) is tuple:
            encoded: list[object] = []
            for nested in raw:
                if type(nested) is replay_v2.ActualReplayRequiredEvidenceV2:
                    encoded.append(
                        {
                            name: getattr(nested, name)
                            for name in REQUIRED_EVIDENCE_FIELDS
                        }
                    )
                elif type(nested) is replay_v2.ActualReplayGateInputDefinitionV2:
                    encoded.append(
                        {
                            name: (
                                getattr(nested, name).value
                                if isinstance(getattr(nested, name), Enum)
                                else getattr(nested, name)
                            )
                            for name in GATE_DEFINITION_FIELDS
                        }
                    )
                else:
                    encoded.append(nested)
            mapping[item.name] = encoded
        else:
            mapping[item.name] = raw
    return mapping


def _install_forbidden_content_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise _ForbiddenBoundaryReached(
            "malformed final supplied item reached content finalization"
        )

    monkeypatch.setattr(replay_v2, "canonical_json", forbidden)
    monkeypatch.setattr(replay_v2, "stable_hash", forbidden)
    monkeypatch.setattr(replay_v2.hashlib, "sha256", forbidden)
    monkeypatch.setattr(
        scoring_v2,
        "build_formal_unsealed_answer_manifest_v2",
        forbidden,
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

    calls = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in (qualified_name(node.func),)
        if name is not None
    }
    return frozenset(imported), frozenset(calls)


def _is_forbidden_import(name: str) -> bool:
    """Match a forbidden module at any complete dotted-path boundary."""

    framed = f".{name}."
    return any(
        f".{forbidden}." in framed
        for forbidden in FORBIDDEN_IMPORT_SUFFIXES
    )


@dataclass(frozen=True, slots=True)
class _SyntheticFixtureV2:
    answer: scoring_v2.FormalUnsealedAnswerManifestV2
    main_row_ids: tuple[str, ...]
    semantic_conflict_row_ids: tuple[str, ...]
    cells: tuple[
        tuple[
            CanonicalFamilyId,
            str,
            tuple[tuple[Phase2BCaseType, MarginStratum], ...],
        ],
        ...,
    ]


def _assert_fixture_oracle(fixture: _SyntheticFixtureV2) -> None:
    assert len(fixture.main_row_ids) == 720
    assert len(fixture.semantic_conflict_row_ids) == 240
    assert fixture.main_row_ids == tuple(sorted(fixture.main_row_ids))
    assert fixture.semantic_conflict_row_ids == tuple(
        sorted(fixture.semantic_conflict_row_ids)
    )
    assert not set(fixture.main_row_ids).intersection(
        fixture.semantic_conflict_row_ids
    )
    assert len(fixture.cells) == 12
    assert len({(family, scale) for family, scale, _ in fixture.cells}) == 12
    observed_case_totals = {case_type: 0 for case_type in Phase2BCaseType}
    for family_id, scale_slice, assignments in fixture.cells:
        assert type(family_id) is CanonicalFamilyId
        assert scale_slice in {"S01", "S02"}
        assert len(assignments) == 60
        for case_type, expected_count in EXPECTED_CASE_QUOTA_PER_CELL.items():
            assert sum(item[0] is case_type for item in assignments) == expected_count
            observed_case_totals[case_type] += expected_count
        for margin, expected_count in EXPECTED_MARGIN_QUOTA_PER_CELL.items():
            assert sum(item[1] is margin for item in assignments) == expected_count
        nonunique = tuple(
            case_type
            for case_type, margin in assignments
            if margin is MarginStratum.NONUNIQUE_OR_INSUFFICIENT
        )
        assert nonunique.count(
            Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE
        ) == 1
        assert nonunique.count(
            Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE
        ) == 8
    assert observed_case_totals == {
        case_type: count * 12
        for case_type, count in EXPECTED_CASE_QUOTA_PER_CELL.items()
    }
    rows = fixture.answer.main_answer_rows
    assert len(rows) == 720
    assert tuple(row.input_row_id for row in rows) == fixture.main_row_ids
    assert len({row.answer_row_id for row in rows}) == 720
    assert all(
        (row.canonical_family_id is not None)
        is (row.case_type in ANSWERABLE_CASE_TYPES)
        for row in rows
    )


def _gate_rows(
    fixture: _SyntheticFixtureV2,
) -> tuple[replay_v2.FormalUnsealedGateInputRowV2, ...]:
    rows: list[replay_v2.FormalUnsealedGateInputRowV2] = []
    index = 0
    for family_id, scale_slice, assignments in fixture.cells:
        scale_id = replay_v2.FormalUnsealedScaleSliceIdV2(scale_slice)
        for case_type, margin_stratum in assignments:
            answer = fixture.answer.main_answer_rows[index]
            rows.append(
                replay_v2.FormalUnsealedGateInputRowV2(
                    input_row_id=answer.input_row_id,
                    answer_row_id=answer.answer_row_id,
                    case_type=case_type,
                    margin_stratum=margin_stratum,
                    canonical_family_id=family_id,
                    scale_slice_id=scale_id,
                    latent_base_case_id=_hex_id(
                        "phase2b_latent_base_case_v2_", index
                    ),
                )
            )
            index += 1
    return tuple(rows)


@pytest.fixture(scope="module")
def synthetic_fixture_v2() -> _SyntheticFixtureV2:
    fixture = _synthetic_fixture_base()
    _assert_fixture_oracle(fixture)
    return fixture


@pytest.fixture(scope="module")
def gate_manifest_v2(
    synthetic_fixture_v2: _SyntheticFixtureV2,
) -> replay_v2.FormalUnsealedGateInputManifestV2:
    return replay_v2.build_formal_unsealed_gate_input_manifest_v2(
        answer_manifest=synthetic_fixture_v2.answer,
        main_gate_input_rows=_gate_rows(synthetic_fixture_v2),
    )


def _validate_kwargs(
    fixture: _SyntheticFixtureV2,
    manifest: replay_v2.FormalUnsealedGateInputManifestV2,
    **changes: object,
) -> dict[str, object]:
    salt = "synthetic-gate-input-opening-salt-0123456789abcdef"
    commitment = replay_v2.salted_gate_input_commitment_sha256_v2(
        manifest.gate_input_manifest_sha256,
        salt,
    )
    values: dict[str, object] = {
        "gate_input_manifest": manifest,
        "answer_manifest": fixture.answer,
        "revealed_gate_input_manifest_sha256": (
            manifest.gate_input_manifest_sha256
        ),
        "gate_input_commitment_salt": salt,
        "salted_gate_input_commitment_sha256": commitment,
    }
    values.update(changes)
    return values


def _assert_atomic_rejection(
    value: object,
    reason: replay_v2.ActualUnsealed960ReplayInputReasonV2 | None = None,
) -> None:
    assert type(value) is replay_v2.ActualUnsealed960ReplayInputContractRejectionV2
    assert value.disposition is (  # type: ignore[attr-defined]
        replay_v2.ActualUnsealed960ReplayInputDispositionV2.REJECTED
    )
    if reason is not None:
        assert value.reason is reason  # type: ignore[attr-defined]
    assert value.validation is None  # type: ignore[attr-defined]
    assert value.required_evidence_inventory == ()  # type: ignore[attr-defined]
    assert value.available_overall_gate_input_definitions == ()  # type: ignore[attr-defined]
    assert value.unavailable_overall_gate_input_definitions == ()  # type: ignore[attr-defined]
    assert value.slice_gate_input_definitions == ()  # type: ignore[attr-defined]
    assert value.metric_results == ()  # type: ignore[attr-defined]
    assert value.scored_rows == ()  # type: ignore[attr-defined]
    assert value.gate_results == ()  # type: ignore[attr-defined]
    assert value.scale_regret_result is None  # type: ignore[attr-defined]
    assert value.bootstrap_result is None  # type: ignore[attr-defined]
    assert value.partial_output_published is False  # type: ignore[attr-defined]
    for name in (*TRUE_RESULT_CLAIMS, *FALSE_RESULT_CLAIMS):
        assert getattr(value, name) is False


def test_public_surface_signatures_fields_and_identity_are_exact() -> None:
    assert replay_v2.__all__ == EXPECTED_PUBLIC_SURFACE
    build = inspect.signature(replay_v2.build_formal_unsealed_gate_input_manifest_v2)
    validate = inspect.signature(
        replay_v2.validate_actual_unsealed_960_replay_input_contract_v2
    )
    assert tuple(build.parameters) == ("answer_manifest", "main_gate_input_rows")
    assert tuple(validate.parameters) == (
        "gate_input_manifest",
        "answer_manifest",
        "revealed_gate_input_manifest_sha256",
        "gate_input_commitment_salt",
        "salted_gate_input_commitment_sha256",
    )
    assert all(
        item.kind is inspect.Parameter.KEYWORD_ONLY
        for signature in (build, validate)
        for item in signature.parameters.values()
    )
    field_expectations = (
        (replay_v2.FormalUnsealedGateInputRowV2, GATE_ROW_FIELDS),
        (replay_v2.ActualReplayGateInputDefinitionV2, GATE_DEFINITION_FIELDS),
        (replay_v2.ActualReplayRequiredEvidenceV2, REQUIRED_EVIDENCE_FIELDS),
        (replay_v2.FormalUnsealedGateInputManifestV2, MANIFEST_FIELDS),
        (replay_v2.ActualUnsealed960ReplayInputContractV2, RESULT_FIELDS),
        (replay_v2.ActualUnsealed960ReplayInputContractRejectionV2, REJECTION_FIELDS),
    )
    for value_type, expected in field_expectations:
        assert tuple(item.name for item in fields(value_type)) == expected
    assert tuple(replay_v2.FormalUnsealedScaleSliceIdV2) == (
        replay_v2.FormalUnsealedScaleSliceIdV2.S01,
        replay_v2.FormalUnsealedScaleSliceIdV2.S02,
    )
    assert replay_v2.ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_VERSION == (
        "hegel-machine-phase2b-actual-unsealed-960-replay-input-contract/2"
    )
    assert replay_v2.ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL == (
        "NON_AUTHORITATIVE_ACTUAL_REPLAY_INPUT_CONTRACT_ONLY"
    )
    assert replay_v2.FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_VERSION == (
        "hegel-machine-phase2b-formal-unsealed-gate-input-manifest/2"
    )
    expected_ids = {
        "ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_SCHEMA_ID": (
            "phase2b_actual_unsealed_960_replay_input_contract_schema_v2_"
            "a4f61ddfb07643e23ac404616062127e2ae6ca02f13b29c265062d6a1f660f4a"
        ),
        "ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_POLICY_ID": (
            "phase2b_actual_unsealed_960_replay_input_contract_policy_v2_"
            "a12ca51dd6f17f29a28a7229f4108c32f438d4775a367a7ad3e5a6275557b531"
        ),
        "FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_ID": (
            "phase2b_formal_unsealed_gate_input_manifest_schema_v2_"
            "7cae322b25caf6e6a9a6239d9ad52281b1e8c7fcc3078b7a72f279de942a19f1"
        ),
        "FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_POLICY_ID": (
            "phase2b_formal_unsealed_gate_input_manifest_policy_v2_"
            "5168476f2f5483a90c75f4cebc05f4210c378333c2993b4973949d8a32e9aa9a"
        ),
    }
    for name, prefix in (
        (
            "ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_SCHEMA_ID",
            "phase2b_actual_unsealed_960_replay_input_contract_schema_v2_",
        ),
        (
            "ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_POLICY_ID",
            "phase2b_actual_unsealed_960_replay_input_contract_policy_v2_",
        ),
        (
            "FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_ID",
            "phase2b_formal_unsealed_gate_input_manifest_schema_v2_",
        ),
        (
            "FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_POLICY_ID",
            "phase2b_formal_unsealed_gate_input_manifest_policy_v2_",
        ),
    ):
        value = getattr(replay_v2, name)
        assert type(value) is str
        assert value.startswith(prefix)
        assert len(value) == len(prefix) + 64
        assert value == expected_ids[name]


def test_upstream_answer_field_manifests_are_locally_frozen_and_exact() -> None:
    assert tuple(
        item.name for item in fields(scoring_v2.FormalUnsealedAnswerRowV2)
    ) == UPSTREAM_ANSWER_ROW_FIELDS
    assert tuple(
        item.name for item in fields(scoring_v2.FormalUnsealedAnswerManifestV2)
    ) == UPSTREAM_ANSWER_MANIFEST_FIELDS


def test_gate_commitment_is_domain_separated_from_answer_commitment() -> None:
    digest = hashlib.sha256(b"same supplied digest").hexdigest()
    salt = "same-sufficiently-long-synthetic-opening-salt-0123456789"
    encoded_salt = salt.encode("utf-8")
    raw_manifest_digest = bytes.fromhex(digest)
    independent_preimage = (
        b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_COMMITMENT/V2\x00"
        + len(encoded_salt).to_bytes(4, "big")
        + encoded_salt
        + raw_manifest_digest
    )
    independently_reconstructed = hashlib.sha256(independent_preimage).hexdigest()
    assert len(encoded_salt) == 56
    assert len(raw_manifest_digest) == 32
    assert independently_reconstructed == (
        "ef38bdf4559953aec0c60c4dac344f30e9cdf407568a61fe87683a27b610fad5"
    )
    gate_commitment = replay_v2.salted_gate_input_commitment_sha256_v2(
        digest,
        salt,
    )
    assert gate_commitment == independently_reconstructed
    assert gate_commitment != salted_answer_commitment_sha256(digest, salt)
    assert replay_v2.salted_gate_input_commitment_sha256_v2(digest, salt) == (
        gate_commitment
    )


def test_private_outputs_are_not_publicly_constructible() -> None:
    for value_type in (
        replay_v2.ActualReplayGateInputDefinitionV2,
        replay_v2.ActualReplayRequiredEvidenceV2,
        replay_v2.FormalUnsealedGateInputManifestV2,
        replay_v2.ActualUnsealed960ReplayInputContractV2,
        replay_v2.ActualUnsealed960ReplayInputContractRejectionV2,
    ):
        with pytest.raises(TypeError, match="privately issued"):
            value_type()  # type: ignore[call-arg]


def test_gate_manifest_is_complete_content_addressed_and_deterministic(
    synthetic_fixture_v2: _SyntheticFixtureV2,
    gate_manifest_v2: replay_v2.FormalUnsealedGateInputManifestV2,
) -> None:
    rows = gate_manifest_v2.main_gate_input_rows
    assert len(rows) == 720
    assert tuple(row.input_row_id for row in rows) == synthetic_fixture_v2.main_row_ids
    assert tuple(row.answer_row_id for row in rows) == tuple(
        row.answer_row_id for row in synthetic_fixture_v2.answer.main_answer_rows
    )
    assert tuple(row.case_type for row in rows) == tuple(
        row.case_type for row in synthetic_fixture_v2.answer.main_answer_rows
    )
    assert len({row.gate_input_row_id for row in rows}) == 720
    assert len({row.latent_base_case_id for row in rows}) == 720
    assert all(
        row.gate_input_row_id.startswith(
            "phase2b_actual_replay_gate_input_row_v2_"
        )
        for row in rows
    )
    assert gate_manifest_v2.main_gate_input_row_ids_root.startswith(
        "phase2b_actual_replay_gate_input_rows_v2_"
    )
    assert gate_manifest_v2.gate_input_manifest_id == (
        "phase2b_formal_unsealed_gate_input_manifest_v2_"
        + gate_manifest_v2.gate_input_manifest_sha256
    )
    assert gate_manifest_v2.schema_version == (
        replay_v2.FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_VERSION
    )
    assert gate_manifest_v2.schema_id == (
        replay_v2.FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_SCHEMA_ID
    )
    assert gate_manifest_v2.policy_id == (
        replay_v2.FORMAL_UNSEALED_GATE_INPUT_MANIFEST_V2_POLICY_ID
    )
    assert gate_manifest_v2.claim_level == (
        replay_v2.ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL
    )
    assert gate_manifest_v2.answer_manifest_id == (
        synthetic_fixture_v2.answer.answer_manifest_id
    )
    assert gate_manifest_v2.answer_manifest_sha256 == (
        synthetic_fixture_v2.answer.answer_manifest_sha256
    )
    assert gate_manifest_v2.main_answer_row_ids_root == (
        synthetic_fixture_v2.answer.main_answer_row_ids_root
    )
    assert gate_manifest_v2.required_evidence_inventory
    assert all(
        item.supplied_by_this_contract is False
        and item.verifier_implemented is False
        for item in gate_manifest_v2.required_evidence_inventory
    )
    assert len({item.requirement_id for item in gate_manifest_v2.required_evidence_inventory}) == len(
        gate_manifest_v2.required_evidence_inventory
    )
    rebuilt = replay_v2.build_formal_unsealed_gate_input_manifest_v2(
        answer_manifest=synthetic_fixture_v2.answer,
        main_gate_input_rows=_gate_rows(synthetic_fixture_v2),
    )
    assert rebuilt == gate_manifest_v2
    assert rebuilt is not gate_manifest_v2
    assert rebuilt.main_gate_input_rows is not rows
    assert all(left is not right for left, right in zip(rebuilt.main_gate_input_rows, rows))
    assert rebuilt.required_evidence_inventory is not (
        gate_manifest_v2.required_evidence_inventory
    )
    assert all(
        left is not right
        for left, right in zip(
            rebuilt.required_evidence_inventory,
            gate_manifest_v2.required_evidence_inventory,
        )
    )


def test_independent_content_address_oracle_covers_rows_manifest_and_result(
    synthetic_fixture_v2: _SyntheticFixtureV2,
    gate_manifest_v2: replay_v2.FormalUnsealedGateInputManifestV2,
) -> None:
    first = gate_manifest_v2.main_gate_input_rows[0]
    expected_first_id = (
        "phase2b_actual_replay_gate_input_row_v2_"
        + hashlib.sha256(
            b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_ROW/V2\x00"
            + canonical_json(
                _independent_gate_row_mapping(first, include_row_id=False)
            ).encode("utf-8")
        ).hexdigest()
    )
    assert first.gate_input_row_id == expected_first_id

    expected_row_root = _row_sequence_root(
        tuple(
            row.gate_input_row_id
            for row in gate_manifest_v2.main_gate_input_rows
        ),
        domain=b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_ROW_IDS/V2\x00",
        prefix="phase2b_actual_replay_gate_input_rows_v2_",
    )
    assert gate_manifest_v2.main_gate_input_row_ids_root == expected_row_root

    expected_manifest_sha = hashlib.sha256(
        b"HEGEL/PHASE2B/ACTUAL_REPLAY/GATE_INPUT_MANIFEST/V2\x00"
        + canonical_json(
            _independent_manifest_preimage(gate_manifest_v2)
        ).encode("utf-8")
    ).hexdigest()
    assert gate_manifest_v2.gate_input_manifest_sha256 == expected_manifest_sha
    assert gate_manifest_v2.gate_input_manifest_id == (
        "phase2b_formal_unsealed_gate_input_manifest_v2_"
        + expected_manifest_sha
    )

    result = replay_v2.validate_actual_unsealed_960_replay_input_contract_v2(
        **_validate_kwargs(synthetic_fixture_v2, gate_manifest_v2)  # type: ignore[arg-type]
    )
    assert type(result) is replay_v2.ActualUnsealed960ReplayInputContractV2
    expected_result_id = (
        "phase2b_actual_unsealed_960_replay_input_contract_v2_"
        + hashlib.sha256(
            b"HEGEL/PHASE2B/ACTUAL_REPLAY/RESULT/V2\x00"
            + canonical_json(_independent_result_preimage(result)).encode("utf-8")
        ).hexdigest()
    )
    assert result.result_id == expected_result_id


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_container",
        "missing",
        "extra",
        "reordered",
        "duplicate_row",
        "input_row",
        "answer_row",
        "case_type",
        "margin_type",
        "nonunique_margin",
        "family_type",
        "family_mismatch",
        "scale_type",
        "latent_duplicate",
        "latent_prefix",
        "preissued_row_id",
    ),
)
def test_builder_rejects_row_coverage_crossbinding_and_quota_mutations(
    synthetic_fixture_v2: _SyntheticFixtureV2,
    mutation: str,
) -> None:
    rows = _gate_rows(synthetic_fixture_v2)
    supplied: object = rows
    if mutation == "wrong_container":
        supplied = list(rows)
    elif mutation == "missing":
        supplied = rows[:-1]
    elif mutation == "extra":
        supplied = (*rows, rows[-1])
    elif mutation == "reordered":
        supplied = (rows[1], rows[0], *rows[2:])
    elif mutation == "duplicate_row":
        supplied = (rows[0], rows[0], *rows[2:])
    elif mutation == "input_row":
        supplied = (
            *rows[:-1],
            _unchecked_copy(
                rows[-1],
                input_row_id=_hex_id("phase2b_recognizer_input_row_v2_", 9_001),
            ),
        )
    elif mutation == "answer_row":
        supplied = (
            *rows[:-1],
            _unchecked_copy(
                rows[-1],
                answer_row_id=_hex_id(
                    "phase2b_formal_unsealed_answer_row_v2_", 9_002
                ),
            ),
        )
    elif mutation == "case_type":
        supplied = (
            _unchecked_copy(
                rows[0], case_type=Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE
            ),
            *rows[1:],
        )
    elif mutation == "margin_type":
        supplied = (_unchecked_copy(rows[0], margin_stratum="clear_interior"), *rows[1:])
    elif mutation == "nonunique_margin":
        supplied = (
            _unchecked_copy(
                rows[0],
                margin_stratum=MarginStratum.NONUNIQUE_OR_INSUFFICIENT,
            ),
            *rows[1:],
        )
    elif mutation == "family_type":
        supplied = (_unchecked_copy(rows[0], canonical_family_id=rows[0].canonical_family_id.value), *rows[1:])
    elif mutation == "family_mismatch":
        supplied = (
            _unchecked_copy(rows[0], canonical_family_id=CanonicalFamilyId.F02),
            *rows[1:],
        )
    elif mutation == "scale_type":
        supplied = (_unchecked_copy(rows[0], scale_slice_id="S01"), *rows[1:])
    elif mutation == "latent_duplicate":
        supplied = (*rows[:-1], _unchecked_copy(rows[-1], latent_base_case_id=rows[0].latent_base_case_id))
    elif mutation == "latent_prefix":
        supplied = (*rows[:-1], _unchecked_copy(rows[-1], latent_base_case_id=f"{9_003:064x}"))
    elif mutation == "preissued_row_id":
        supplied = (
            _unchecked_copy(
                rows[0],
                gate_input_row_id=_hex_id(
                    "phase2b_actual_replay_gate_input_row_v2_", 9_004
                ),
            ),
            *rows[1:],
        )
    with pytest.raises((TypeError, ValueError)):
        replay_v2.build_formal_unsealed_gate_input_manifest_v2(
            answer_manifest=synthetic_fixture_v2.answer,
            main_gate_input_rows=supplied,  # type: ignore[arg-type]
        )


def test_positive_contract_freezes_inputs_but_no_execution_or_results(
    synthetic_fixture_v2: _SyntheticFixtureV2,
    gate_manifest_v2: replay_v2.FormalUnsealedGateInputManifestV2,
) -> None:
    kwargs = _validate_kwargs(synthetic_fixture_v2, gate_manifest_v2)
    result = replay_v2.validate_actual_unsealed_960_replay_input_contract_v2(
        **kwargs  # type: ignore[arg-type]
    )
    assert type(result) is replay_v2.ActualUnsealed960ReplayInputContractV2
    assert result.disposition is (
        replay_v2.ActualUnsealed960ReplayInputDispositionV2
        .ACTUAL_REPLAY_CONTRACT_COMPLETE_NOT_EXECUTED
    )
    assert result.reason is (
        replay_v2.ActualUnsealed960ReplayInputReasonV2
        .SUPPLIED_720_MAIN_GATE_LABEL_PACKAGE_BOUND_NOT_EXECUTED
    )
    assert result.version == (
        replay_v2.ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_VERSION
    )
    assert result.schema_id == (
        replay_v2.ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_SCHEMA_ID
    )
    assert result.policy_id == (
        replay_v2.ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_POLICY_ID
    )
    assert result.claim_level == (
        replay_v2.ACTUAL_UNSEALED_960_REPLAY_INPUT_CONTRACT_V2_CLAIM_LEVEL
    )
    assert result.result_id.startswith(
        "phase2b_actual_unsealed_960_replay_input_contract_v2_"
    )
    assert result.gate_input_manifest_id == gate_manifest_v2.gate_input_manifest_id
    assert result.gate_input_manifest_sha256 == (
        gate_manifest_v2.gate_input_manifest_sha256
    )
    assert result.salted_gate_input_commitment_sha256 == (
        kwargs["salted_gate_input_commitment_sha256"]
    )
    assert result.gate_input_manifest_schema_id == gate_manifest_v2.schema_id
    assert result.gate_input_manifest_policy_id == gate_manifest_v2.policy_id
    assert result.answer_manifest_id == synthetic_fixture_v2.answer.answer_manifest_id
    assert result.answer_manifest_sha256 == (
        synthetic_fixture_v2.answer.answer_manifest_sha256
    )
    for name in (
        "execution_freeze_manifest_id",
        "input_archive_id",
        "input_archive_sha256",
        "input_archive_version",
        "input_archive_policy_id",
        "batch_id",
        "batch_policy_id",
        "exact_freeze_id",
        "ordered_archive_input_row_ids_root",
        "main_row_ids_root",
        "semantic_conflict_row_ids_root",
        "partition_union_row_ids_root",
        "main_answer_row_ids_root",
    ):
        assert getattr(result, name) == getattr(gate_manifest_v2, name)
    assert result.protocol_id == gate_manifest_v2.phase2b_protocol_id
    assert result.formal_scoring_contract_id == (
        gate_manifest_v2.formal_scoring_contract_id
    )
    assert result.main_gate_input_row_ids_root == (
        gate_manifest_v2.main_gate_input_row_ids_root
    )
    assert result.main_row_count == 720
    assert result.semantic_conflict_expected_row_count == 240
    assert result.total_expected_prediction_count == 960
    assert result.unique_latent_base_case_id_count == 720
    assert result.family_scale_cell_count == 12
    for name in TRUE_RESULT_CLAIMS:
        assert getattr(result, name) is True
    for name in FALSE_RESULT_CLAIMS:
        assert getattr(result, name) is False
    assert result.required_evidence_inventory == (
        gate_manifest_v2.required_evidence_inventory
    )
    assert result.required_evidence_inventory is not (
        gate_manifest_v2.required_evidence_inventory
    )
    assert all(
        left is not right
        for left, right in zip(
            result.required_evidence_inventory,
            gate_manifest_v2.required_evidence_inventory,
        )
    )
    available = result.available_overall_gate_input_definitions
    unavailable = result.unavailable_overall_gate_input_definitions
    slices = result.slice_gate_input_definitions
    assert tuple(item.gate_name for item in available) == EXPECTED_OVERALL_AVAILABLE
    assert tuple(item.gate_name for item in unavailable) == EXPECTED_OVERALL_UNAVAILABLE
    assert all(
        item.input_available is True
        and item.expected_denominator is not None
        and item.missing_input_reason is None
        for item in available
    )
    assert all(
        item.input_available is False
        and item.expected_denominator is None
        and item.missing_input_reason is not None
        for item in unavailable
    )
    assert tuple(
        (item.gate_name, item.scope, item.expected_denominator)
        for item in slices
    ) == EXPECTED_SLICE_DENOMINATORS
    assert all(
        item.input_available is True and item.missing_input_reason is None
        for item in slices
    )
    assert len(
        {
            item.definition_id
            for item in (*available, *unavailable, *slices)
        }
    ) == 18
    assert result.metric_results == ()
    assert result.scored_rows == ()
    assert result.gate_results == ()
    assert result.scale_regret_result is None
    assert result.bootstrap_result is None


def test_available_and_unavailable_gate_rules_are_exact(
    synthetic_fixture_v2: _SyntheticFixtureV2,
    gate_manifest_v2: replay_v2.FormalUnsealedGateInputManifestV2,
) -> None:
    result = replay_v2.validate_actual_unsealed_960_replay_input_contract_v2(
        **_validate_kwargs(synthetic_fixture_v2, gate_manifest_v2)  # type: ignore[arg-type]
    )
    assert type(result) is replay_v2.ActualUnsealed960ReplayInputContractV2
    observed = {
        item.gate_name: (item.expected_denominator, item.success_rule)
        for item in result.available_overall_gate_input_definitions
    }
    assert observed == {
        "family_exact": (240, "positive_decision_and_canonical_family_exact"),
        "binding_exact": (240, "positive_decision_and_canonical_binding_tuple_exact"),
        "scale_set_accuracy": (240, "positive_decision_and_canonical_scale_tuple_exact"),
        "joint_exact": (
            240,
            "expected_decision_family_binding_and_scale_all_exact_set_rows_require_answer_set",
        ),
        "hard_negative_rejection": (96, "predicted_decision_is_abstain_reason_unscored"),
        "binding_counterfactual_rejection": (
            96,
            "predicted_decision_is_abstain_reason_unscored",
        ),
        "scale_counterfactual_rejection": (
            96,
            "predicted_decision_is_abstain_reason_unscored",
        ),
        "sign_or_invariant_break_rejection": (
            96,
            "predicted_decision_is_abstain_reason_unscored",
        ),
        "abstention_specificity": (228, "unique_answerable_decision_is_not_abstain"),
        "nonidentifiable_scale_abstention": (
            96,
            "insufficient_or_nonidentifiable_decision_is_abstain_reason_unscored",
        ),
    }
    unavailable = {
        item.gate_name: (
            item.expected_denominator,
            item.success_rule,
            item.missing_input_reason,
        )
        for item in result.unavailable_overall_gate_input_definitions
    }
    assert unavailable == {
        "fail_closed_rate": (
            None,
            "requires_attempt_event_denominator_not_abstention",
            "durable_attempt_event_manifest_not_supplied",
        ),
        "preservation_consistency": (
            None,
            "requires_detached_496_legal_and_76_invalid_pair_results",
            "preservation_pair_result_manifest_not_supplied",
        ),
    }


@pytest.mark.parametrize(
    ("target", "field", "replacement"),
    (
        ("manifest", "schema_version", "cross-version"),
        ("manifest", "schema_id", "phase2b_formal_unsealed_gate_input_manifest_schema_v2_" + "0" * 64),
        ("manifest", "policy_id", "phase2b_formal_unsealed_gate_input_manifest_policy_v2_" + "0" * 64),
        ("manifest", "claim_level", "ACTUAL_EVIDENCE"),
        ("manifest", "answer_manifest_sha256", "1" * 64),
        ("manifest", "main_row_ids_root", "phase2b_unsealed_main_rows_v2_" + "2" * 64),
        ("manifest", "main_gate_input_row_ids_root", "phase2b_actual_replay_gate_input_rows_v2_" + "3" * 64),
        ("manifest", "gate_input_manifest_sha256", "4" * 64),
        ("manifest", "gate_input_manifest_id", "phase2b_formal_unsealed_gate_input_manifest_v2_" + "5" * 64),
        ("answer", "answer_manifest_sha256", "6" * 64),
        ("answer", "main_answer_row_ids_root", "phase2b_formal_unsealed_answer_rows_v2_" + "7" * 64),
    ),
)
def test_validator_splice_matrix_is_atomic(
    synthetic_fixture_v2: _SyntheticFixtureV2,
    gate_manifest_v2: replay_v2.FormalUnsealedGateInputManifestV2,
    target: str,
    field: str,
    replacement: object,
) -> None:
    manifest = gate_manifest_v2
    answer = synthetic_fixture_v2.answer
    if target == "manifest":
        manifest = _unchecked_copy(manifest, **{field: replacement})  # type: ignore[assignment]
    else:
        answer = _unchecked_copy(answer, **{field: replacement})  # type: ignore[assignment]
    value = replay_v2.validate_actual_unsealed_960_replay_input_contract_v2(
        **_validate_kwargs(
            synthetic_fixture_v2,
            gate_manifest_v2,
            gate_input_manifest=manifest,
            answer_manifest=answer,
        )  # type: ignore[arg-type]
    )
    _assert_atomic_rejection(value)


@pytest.mark.parametrize(
    "changes",
    (
        {"revealed_gate_input_manifest_sha256": "8" * 64},
        {"salted_gate_input_commitment_sha256": "9" * 64},
        {"gate_input_commitment_salt": "too-short"},
        {"gate_input_commitment_salt": "x" * 4097},
        {"gate_input_commitment_salt": 123},
    ),
)
def test_opening_mutations_are_atomic_and_claim_no_timing_or_authority(
    synthetic_fixture_v2: _SyntheticFixtureV2,
    gate_manifest_v2: replay_v2.FormalUnsealedGateInputManifestV2,
    changes: dict[str, object],
) -> None:
    value = replay_v2.validate_actual_unsealed_960_replay_input_contract_v2(
        **_validate_kwargs(synthetic_fixture_v2, gate_manifest_v2, **changes)  # type: ignore[arg-type]
    )
    _assert_atomic_rejection(
        value,
        replay_v2.ActualUnsealed960ReplayInputReasonV2.GATE_INPUT_OPENING_INVALID,
    )


def test_wrong_exact_types_are_atomic(
    synthetic_fixture_v2: _SyntheticFixtureV2,
    gate_manifest_v2: replay_v2.FormalUnsealedGateInputManifestV2,
) -> None:
    for changes in (
        {"gate_input_manifest": object()},
        {"answer_manifest": object()},
        {"revealed_gate_input_manifest_sha256": b"0" * 64},
    ):
        value = replay_v2.validate_actual_unsealed_960_replay_input_contract_v2(
            **_validate_kwargs(synthetic_fixture_v2, gate_manifest_v2, **changes)  # type: ignore[arg-type]
        )
        _assert_atomic_rejection(value)


def test_precommit_schema_excludes_all_downstream_prediction_identifiers() -> None:
    assert replay_v2._FORBIDDEN_DOWNSTREAM_FIELDS_V2 == (
        EXPECTED_FORBIDDEN_DOWNSTREAM_FIELDS
    )
    for value_type in (
        replay_v2.FormalUnsealedGateInputRowV2,
        replay_v2.FormalUnsealedGateInputManifestV2,
    ):
        assert not EXPECTED_FORBIDDEN_DOWNSTREAM_FIELDS.intersection(
            item.name for item in fields(value_type)
        )
    folded_public = " ".join(replay_v2.__all__).casefold()
    for forbidden in (
        "score_predictions",
        "evaluate_gate",
        "run_recognizer",
        "prediction_archive",
    ):
        assert forbidden not in folded_public


def test_source_boundary_has_no_execution_scoring_rng_or_private_upstream_calls() -> None:
    source = Path(replay_v2.__file__).read_text(encoding="utf-8")
    imported, calls = _ast_direct_imports_and_calls(source)
    for name in imported:
        assert not _is_forbidden_import(name), name
    for name in calls:
        terminal = name.rsplit(".", 1)[-1]
        assert terminal not in FORBIDDEN_CALL_TERMINALS, name
    assert "docker" not in source.casefold()
    assert "ledger.transition" not in source.casefold()
    assert "_validate_answer_manifest_v2" not in source
    assert "_answer_manifest_sha_v2" not in source
    assert "_validate_answer_rows_v2" not in source


@pytest.mark.parametrize(
    ("source", "expected_import"),
    (
        ("import concurrent.futures as futures", "concurrent.futures"),
        ("from urllib import request as urlrequest", "urllib.request"),
        (
            "import hegel_machine.phase2b_runner as runner",
            "hegel_machine.phase2b_runner",
        ),
    ),
)
def test_ast_forbidden_import_guard_detects_dotted_paths(
    source: str,
    expected_import: str,
) -> None:
    imported, _calls = _ast_direct_imports_and_calls(source)
    assert expected_import in imported
    assert _is_forbidden_import(expected_import)
    assert not _is_forbidden_import("hashlib")


def test_builder_validates_all_rows_before_any_content_hash(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_fixture_v2: _SyntheticFixtureV2,
) -> None:
    rows = _gate_rows(synthetic_fixture_v2)
    malformed = _unchecked_copy(rows[-1], margin_stratum="near_boundary_identifiable")

    def forbidden_hash(*args: object, **kwargs: object) -> object:
        raise _ForbiddenBoundaryReached("malformed final row reached hashing")

    monkeypatch.setattr(replay_v2, "stable_hash", forbidden_hash)
    monkeypatch.setattr(replay_v2.hashlib, "sha256", forbidden_hash)
    with pytest.raises((TypeError, ValueError)):
        replay_v2.build_formal_unsealed_gate_input_manifest_v2(
            answer_manifest=synthetic_fixture_v2.answer,
            main_gate_input_rows=(*rows[:-1], malformed),
        )


def test_validator_rejects_cheap_identity_before_any_content_hash(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_fixture_v2: _SyntheticFixtureV2,
    gate_manifest_v2: replay_v2.FormalUnsealedGateInputManifestV2,
) -> None:
    malformed = _unchecked_copy(gate_manifest_v2, schema_version="cross-version")
    kwargs = _validate_kwargs(
        synthetic_fixture_v2,
        gate_manifest_v2,
        gate_input_manifest=malformed,
    )

    def forbidden_hash(*args: object, **kwargs: object) -> object:
        raise _ForbiddenBoundaryReached("cheap identity reached hashing")

    monkeypatch.setattr(replay_v2, "stable_hash", forbidden_hash)
    monkeypatch.setattr(replay_v2.hashlib, "sha256", forbidden_hash)
    value = replay_v2.validate_actual_unsealed_960_replay_input_contract_v2(
        **kwargs  # type: ignore[arg-type]
    )
    _assert_atomic_rejection(
        value,
        replay_v2.ActualUnsealed960ReplayInputReasonV2.CROSS_VERSION_INPUT,
    )


def test_validator_checks_malformed_last_row_before_any_content_hash(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_fixture_v2: _SyntheticFixtureV2,
    gate_manifest_v2: replay_v2.FormalUnsealedGateInputManifestV2,
) -> None:
    rows = gate_manifest_v2.main_gate_input_rows
    malformed_row = _unchecked_copy(rows[-1], scale_slice_id="S02")
    malformed_manifest = _unchecked_copy(
        gate_manifest_v2,
        main_gate_input_rows=(*rows[:-1], malformed_row),
    )
    kwargs = _validate_kwargs(
        synthetic_fixture_v2,
        gate_manifest_v2,
        gate_input_manifest=malformed_manifest,
    )

    def forbidden_hash(*args: object, **kwargs: object) -> object:
        raise _ForbiddenBoundaryReached("malformed final row reached hashing")

    monkeypatch.setattr(replay_v2, "stable_hash", forbidden_hash)
    monkeypatch.setattr(replay_v2.hashlib, "sha256", forbidden_hash)
    value = replay_v2.validate_actual_unsealed_960_replay_input_contract_v2(
        **kwargs  # type: ignore[arg-type]
    )
    _assert_atomic_rejection(
        value,
        replay_v2.ActualUnsealed960ReplayInputReasonV2.GATE_INPUT_MANIFEST_INVALID,
    )


def test_validator_checks_malformed_final_answer_row_before_all_hash_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_fixture_v2: _SyntheticFixtureV2,
    gate_manifest_v2: replay_v2.FormalUnsealedGateInputManifestV2,
) -> None:
    answer_rows = synthetic_fixture_v2.answer.main_answer_rows
    valid_binding = RoleBinding(
        role_id=_uuid4(99_000),
        entity_id=_uuid4(99_001),
    )
    malformed_binding = _unchecked_copy(
        valid_binding,
        role_id="not-a-canonical-uuid",
    )
    malformed_last = _unchecked_copy(
        answer_rows[-1],
        binding=(malformed_binding,),
    )
    malformed_answer = _unchecked_copy(
        synthetic_fixture_v2.answer,
        main_answer_rows=(*answer_rows[:-1], malformed_last),
    )
    kwargs = _validate_kwargs(
        synthetic_fixture_v2,
        gate_manifest_v2,
        answer_manifest=malformed_answer,
    )
    _install_forbidden_content_boundaries(monkeypatch)
    value = replay_v2.validate_actual_unsealed_960_replay_input_contract_v2(
        **kwargs  # type: ignore[arg-type]
    )
    _assert_atomic_rejection(
        value,
        replay_v2.ActualUnsealed960ReplayInputReasonV2.ANSWER_MANIFEST_INVALID,
    )


def test_validator_checks_malformed_final_evidence_before_all_hash_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_fixture_v2: _SyntheticFixtureV2,
    gate_manifest_v2: replay_v2.FormalUnsealedGateInputManifestV2,
) -> None:
    inventory = gate_manifest_v2.required_evidence_inventory
    malformed_last = _unchecked_copy(
        inventory[-1],
        verifier_implemented=True,
    )
    malformed_manifest = _unchecked_copy(
        gate_manifest_v2,
        required_evidence_inventory=(*inventory[:-1], malformed_last),
    )
    kwargs = _validate_kwargs(
        synthetic_fixture_v2,
        gate_manifest_v2,
        gate_input_manifest=malformed_manifest,
    )
    _install_forbidden_content_boundaries(monkeypatch)
    value = replay_v2.validate_actual_unsealed_960_replay_input_contract_v2(
        **kwargs  # type: ignore[arg-type]
    )
    _assert_atomic_rejection(
        value,
        replay_v2.ActualUnsealed960ReplayInputReasonV2.GATE_INPUT_MANIFEST_INVALID,
    )


def test_builder_fresh_copies_rows_before_caller_pollution(
    synthetic_fixture_v2: _SyntheticFixtureV2,
) -> None:
    supplied = _gate_rows(synthetic_fixture_v2)
    original_input_id = supplied[0].input_row_id
    manifest = replay_v2.build_formal_unsealed_gate_input_manifest_v2(
        answer_manifest=synthetic_fixture_v2.answer,
        main_gate_input_rows=supplied,
    )
    object.__setattr__(
        supplied[0],
        "input_row_id",
        _hex_id("phase2b_recognizer_input_row_v2_", 99_999),
    )
    assert manifest.main_gate_input_rows[0] is not supplied[0]
    assert manifest.main_gate_input_rows[0].input_row_id == original_input_id


def test_validator_hash_time_caller_mutation_uses_only_closed_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_fixture_v2: _SyntheticFixtureV2,
    gate_manifest_v2: replay_v2.FormalUnsealedGateInputManifestV2,
) -> None:
    answer_rows = tuple(
        _unchecked_copy(row)
        for row in synthetic_fixture_v2.answer.main_answer_rows
    )
    caller_answer = _unchecked_copy(
        synthetic_fixture_v2.answer,
        main_answer_rows=answer_rows,
    )
    gate_rows = tuple(
        _unchecked_copy(row) for row in gate_manifest_v2.main_gate_input_rows
    )
    inventory = tuple(
        _unchecked_copy(item)
        for item in gate_manifest_v2.required_evidence_inventory
    )
    caller_manifest = _unchecked_copy(
        gate_manifest_v2,
        main_gate_input_rows=gate_rows,
        required_evidence_inventory=inventory,
    )
    expected_input_archive_id = gate_manifest_v2.input_archive_id
    expected_answer_manifest_id = gate_manifest_v2.answer_manifest_id
    expected_gate_root = gate_manifest_v2.main_gate_input_row_ids_root
    kwargs = _validate_kwargs(
        synthetic_fixture_v2,
        gate_manifest_v2,
        gate_input_manifest=caller_manifest,
        answer_manifest=caller_answer,
    )
    original_canonical_json = replay_v2.canonical_json
    mutation_count = 0

    def mutate_callers_at_first_hash(value: object) -> str:
        nonlocal mutation_count
        if mutation_count == 0:
            mutation_count += 1
            object.__setattr__(
                caller_answer,
                "input_archive_id",
                _hex_id("phase2b_recognizer_input_archive_v2_", 98_001),
            )
            object.__setattr__(caller_answer, "main_answer_rows", ())
            object.__setattr__(
                caller_manifest,
                "input_archive_id",
                _hex_id("phase2b_recognizer_input_archive_v2_", 98_002),
            )
            object.__setattr__(caller_manifest, "main_gate_input_rows", ())
            object.__setattr__(caller_manifest, "required_evidence_inventory", ())
            object.__setattr__(
                answer_rows[-1],
                "answer_row_id",
                _hex_id("phase2b_formal_unsealed_answer_row_v2_", 98_003),
            )
            object.__setattr__(
                gate_rows[-1],
                "latent_base_case_id",
                _hex_id("phase2b_latent_base_case_v2_", 98_004),
            )
            object.__setattr__(inventory[-1], "purpose", "mutated-after-preflight")
        return original_canonical_json(value)

    monkeypatch.setattr(replay_v2, "canonical_json", mutate_callers_at_first_hash)
    result = replay_v2.validate_actual_unsealed_960_replay_input_contract_v2(
        **kwargs  # type: ignore[arg-type]
    )
    assert mutation_count == 1
    assert type(result) is replay_v2.ActualUnsealed960ReplayInputContractV2
    assert result.input_archive_id == expected_input_archive_id
    assert result.answer_manifest_id == expected_answer_manifest_id
    assert result.main_gate_input_row_ids_root == expected_gate_root
    assert result.gate_input_manifest_id == gate_manifest_v2.gate_input_manifest_id
    assert caller_answer.main_answer_rows == ()
    assert caller_manifest.main_gate_input_rows == ()
    assert caller_manifest.required_evidence_inventory == ()


def _synthetic_fixture_base() -> _SyntheticFixtureV2:
    all_ids = tuple(
        sorted(
            _hex_id("phase2b_recognizer_input_row_v2_", index)
            for index in range(960)
        )
    )
    main_row_ids = all_ids[:720]
    semantic_conflict_row_ids = all_ids[720:]
    families = tuple(CanonicalFamilyId)
    scale_slices = ("S01", "S02")
    cells: list[
        tuple[
            CanonicalFamilyId,
            str,
            tuple[tuple[Phase2BCaseType, MarginStratum], ...],
        ]
    ] = []
    answer_rows: list[scoring_v2.FormalUnsealedAnswerRowV2] = []
    row_index = 0
    for family_id in families:
        for scale_slice in scale_slices:
            identifiable_cases = tuple(
                case_type
                for case_type, count in EXPECTED_CASE_QUOTA_PER_CELL.items()
                if case_type
                not in {
                    Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE,
                    Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE,
                }
                for _ in range(count)
            )
            identifiable_margins = (
                (MarginStratum.CLEAR_INTERIOR,) * 21
                + (MarginStratum.MODERATE,) * 18
                + (MarginStratum.NEAR_BOUNDARY_IDENTIFIABLE,) * 12
            )
            assignments = tuple(zip(identifiable_cases, identifiable_margins, strict=True))
            assignments += (
                (
                    Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE,
                    MarginStratum.NONUNIQUE_OR_INSUFFICIENT,
                ),
            )
            assignments += (
                (
                    Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE,
                    MarginStratum.NONUNIQUE_OR_INSUFFICIENT,
                ),
            ) * 8
            assert len(assignments) == 60
            cells.append((family_id, scale_slice, assignments))
            for case_type, _margin in assignments:
                answer_rows.append(
                    _synthetic_answer_row(
                        index=row_index,
                        input_row_id=main_row_ids[row_index],
                        case_type=case_type,
                        family_id=family_id,
                    )
                )
                row_index += 1
    exact_freeze = frozen_phase2b_exact_freeze()
    protocol = frozen_phase2b_protocol()
    answer = scoring_v2.build_formal_unsealed_answer_manifest_v2(
        input_archive_id=_hex_id("phase2b_recognizer_input_archive_v2_", 1),
        input_archive_sha256=hashlib.sha256(b"synthetic input archive").hexdigest(),
        input_archive_version=TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
        input_archive_policy_id=RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
        batch_id=_hex_id("phase2b_trusted_wire_batch_v2_", 2),
        batch_policy_id=TRUSTED_WIRE_BATCH_V2_POLICY_ID,
        exact_freeze_id=exact_freeze.freeze_id,
        phase2b_protocol_id=protocol.protocol_id,
        execution_freeze_manifest_id=_hex_id("phase2b_execution_freeze_", 3),
        ordered_archive_input_row_ids_root=_ordered_root(all_ids),
        main_row_ids_root=_main_root(main_row_ids),
        semantic_conflict_row_ids_root=_challenge_root(semantic_conflict_row_ids),
        partition_union_row_ids_root=_union_root(all_ids),
        main_answer_rows=tuple(answer_rows),
    )
    return _SyntheticFixtureV2(
        answer=answer,
        main_row_ids=main_row_ids,
        semantic_conflict_row_ids=semantic_conflict_row_ids,
        cells=tuple(cells),
    )
