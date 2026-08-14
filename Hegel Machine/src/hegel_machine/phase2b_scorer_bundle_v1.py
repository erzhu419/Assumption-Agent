"""Bounded execution-free Phase-2B scorer bundle mechanics.

The binary bundle transports one canonical execution-freeze manifest, the
public V2 input and prediction archives, evaluator-side answer/gate rows and
their commitment openings.  Decoding reconstructs every private upstream DTO
through its public builder or validator; it performs no filesystem, Docker,
clock, randomness, signature, authority, or actual-C1 operation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import re
import struct
from types import MappingProxyType
from typing import Final, Mapping, NoReturn

from .phase2b_c1_container_actor_profile_v2 import (
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID,
)
from .phase2b_actual_unsealed_960_replay_input_contract_v2 import (
    ActualUnsealed960ReplayInputContractRejectionV2,
    ActualUnsealed960ReplayInputContractV2,
    FormalUnsealedGateInputRowV2,
    FormalUnsealedGateInputManifestV2,
    FormalUnsealedScaleSliceIdV2,
    build_formal_unsealed_gate_input_manifest_v2,
    salted_gate_input_commitment_sha256_v2,
    validate_actual_unsealed_960_replay_input_contract_v2,
)
from .phase2b_formal_unsealed_prediction_scoring_contract_v2 import (
    FormalUnsealedAnswerRowV2,
    FormalUnsealedAnswerManifestV2,
    FormalUnsealedPredictionScoringContractRejectionV2,
    FormalUnsealedPredictionScoringContractValidationV2,
    build_formal_unsealed_answer_manifest_v2,
    validate_formal_unsealed_prediction_scoring_contract_v2,
)
from .phase2b_freeze_v1 import CanonicalFamilyId
from .phase2b_recognizer_input_archive_v2 import (
    ARCHIVE_HEADER_BYTES_V2,
    MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
    DecodedRecognizerInputArchiveV2,
    decode_public_recognizer_input_archive_v2,
)
from .phase2b_recognizer_prediction_archive_v2 import (
    MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2,
    PREDICTION_ARCHIVE_HEADER_BYTES_V2,
    DecodedRecognizerPredictionArchiveV2,
    decode_public_recognizer_prediction_archive_v2,
)
from .phase2b_recognizer_runtime_cli_v2 import (
    MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2,
    decode_execution_freeze_manifest_json_v2,
)
from .phase2b_protocol import (
    ExecutionFreezeManifest,
    MarginStratum,
    Phase2BCaseType,
)
from .phase2b_recognizer_prediction_v2 import PredictionDecisionV2
from .phase2b_strict_recognizer_cli_v2 import (
    StrictRecognizerStructuralReceiptV2,
    verify_strict_recognizer_io_structure_v2,
)
from .phase2b_unsealed_960_available_gate_mechanics_v2 import (
    Unsealed960AvailableGateMechanicsRejectionV2,
    Unsealed960AvailableGateMechanicsV2,
    evaluate_unsealed_960_available_gate_mechanics_v2,
)
from .phase2b_unsealed_960_prediction_scoring_mechanics_v2 import (
    Unsealed960PredictionScoringRejectionV2,
    Unsealed960PredictionScoringMechanicsV2,
    score_unsealed_960_prediction_scoring_mechanics_v2,
)
from .phase2b_unsealed_prediction_evaluator_v2 import (
    UnsealedPredictionEvaluationRejectionV2,
    UnsealedPredictionPartitionManifestV2,
    UnsealedPredictionStructuralEvaluationV2,
    build_unsealed_prediction_partition_manifest_v2,
    evaluate_unsealed_prediction_archive_structure_v2,
)
from .phase2b_wire import RoleBinding


PHASE2B_SCORER_BUNDLE_V1_FILENAME: Final = "phase2b_scorer_bundle_v1.bin"
PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME: Final = (
    "score_mechanics_report_v1.json"
)
PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME: Final = (
    "available_gate_mechanics_report_v1.json"
)
PHASE2B_SCORER_BUNDLE_V1_VERSION: Final = "hegel-phase2b-scorer-bundle/1"
PHASE2B_SCORER_BUNDLE_V1_MAGIC: Final = b"HEGEL-PHASE2B-SCORER-BUNDLE-V1\x00\x00"
PHASE2B_SCORING_SOURCE_SHA256_V1: Final = (
    "c3c12e9cd72a930b9ca8667aab22c4eb3675ceab6249b0b26ba940bf794e7f11"
)
PHASE2B_SCORER_ACTOR_PROFILE_CONTENT_ID_V1: Final = (
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
)
PHASE2B_SCORER_ACTOR_PROFILE_CANONICAL_SHA256_V1: Final = (
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
)
MAXIMUM_PHASE2B_SCORER_BUNDLE_BYTES_V1: Final = 512 * 1024 * 1024
MAXIMUM_PHASE2B_SCORER_METADATA_BYTES_V1: Final = 8 * 1024 * 1024

# Compact aliases retained for the fixed runtime/host surface.
SCORER_BUNDLE_FILENAME_V1: Final = PHASE2B_SCORER_BUNDLE_V1_FILENAME
SCORE_MECHANICS_REPORT_FILENAME_V1: Final = (
    PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME
)
AVAILABLE_GATE_MECHANICS_REPORT_FILENAME_V1: Final = (
    PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME
)
MAXIMUM_SCORER_BUNDLE_BYTES_V1: Final = MAXIMUM_PHASE2B_SCORER_BUNDLE_BYTES_V1

_HEADER_V1: Final = struct.Struct(">32sQQQQ")
_BUNDLE_ID_PREFIX_V1: Final = "phase2b_scorer_bundle_v1_"
_RUN_MANIFEST_ID_PREFIX_V1: Final = "phase2b_scorer_run_manifest_v1_"
_RUN_MANIFEST_DOMAIN_V1: Final = b"HEGEL/PHASE2B/SCORER/RUN_MANIFEST/V1\x00"
SCORE_REPORT_ID_PREFIX_V1: Final = "phase2b_score_mechanics_report_v1_"
SCORE_REPORT_DOMAIN_V1: Final = b"HEGEL/PHASE2B/SCORER/SCORE_REPORT/V1\x00"
GATE_REPORT_ID_PREFIX_V1: Final = "phase2b_available_gate_mechanics_report_v1_"
GATE_REPORT_DOMAIN_V1: Final = b"HEGEL/PHASE2B/SCORER/GATE_REPORT/V1\x00"
PHASE2B_C1_OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V1: Final = (
    "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
    "53631d563f8da8f26c7202b7dd0681be5423fe2ca5c15e9a9f5deaa2920d9940"
)
_EXACT_MAIN_COUNT_V1: Final = 720
_EXACT_CHALLENGE_COUNT_V1: Final = 240
_MAXIMUM_SALT_BYTES_V1: Final = 4096
_MAXIMUM_ID_BYTES_V1: Final = 512

_FALSE_CLAIMS_V1: Final = (
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

_TOP_LEVEL_METADATA_FIELDS_V1: Final = (
    "answer_opening",
    "answer_rows",
    "claims",
    "gate_opening",
    "gate_rows",
    "main_row_ids",
    "run_manifest",
    "schema_version",
    "sections",
    "semantic_conflict_row_ids",
)
_SECTION_FIELDS_V1: Final = ("length", "sha256")
_SECTION_NAMES_V1: Final = (
    "execution_freeze",
    "input_archive",
    "prediction_archive",
)
_OPENING_FIELDS_V1: Final = (
    "commitment_sha256",
    "manifest_sha256",
    "salt",
)
_ANSWER_ROW_FIELDS_V1: Final = (
    "admissible_scale_ids",
    "binding",
    "canonical_family_id",
    "case_type",
    "expected_decision",
    "input_row_id",
)
_GATE_ROW_FIELDS_V1: Final = (
    "canonical_family_id",
    "case_type",
    "input_row_id",
    "latent_base_case_id",
    "margin_stratum",
    "scale_slice_id",
)
_RUN_FIELDS_WITHOUT_ID_V1: Final = (
    "authority_requirement_id",
    "actor_profile_content_id",
    "actor_profile_canonical_sha256",
    "campaign_id",
    "attempt_id",
    "execution_registry_id",
    "scorer_image_digest",
    "scorer_source_root_sha256",
    "protocol_id",
    "exact_freeze_id",
    "execution_freeze_manifest_id",
    "input_archive_id",
    "input_archive_sha256",
    "prediction_archive_id",
    "prediction_archive_sha256",
    "answer_commitment_sha256",
    "gate_input_commitment_sha256",
    "prediction_if_absent_receipt_id",
    "signed_reveal_token_id",
    "scoring_source_sha256",
)
_RUN_FIELDS_V1: Final = (*_RUN_FIELDS_WITHOUT_ID_V1, "run_manifest_id")


class Phase2BScorerBundleErrorV1(ValueError):
    """Stable fail-closed bundle error."""


def _fail(detail: str) -> NoReturn:
    raise Phase2BScorerBundleErrorV1(detail)


def _canonical_json_bytes_v1(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        _fail(f"canonical JSON encoding failed: {exc}")


def _unique_object_v1(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("metadata JSON repeats an object key")
        result[key] = value
    return result


def _reject_constant_v1(value: str) -> NoReturn:
    _fail(f"metadata JSON constant {value!r} is forbidden")


def _reject_float_v1(value: str) -> NoReturn:
    _fail(f"metadata JSON float {value!r} is forbidden")


def _decode_canonical_json_v1(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes or not (
        1 <= len(payload) <= MAXIMUM_PHASE2B_SCORER_METADATA_BYTES_V1
    ):
        _fail("metadata byte bound differs")
    try:
        raw = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_unique_object_v1,
            parse_constant=_reject_constant_v1,
            parse_float=_reject_float_v1,
        )
    except Phase2BScorerBundleErrorV1:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        _fail(f"metadata JSON is invalid: {exc}")
    if type(raw) is not dict:
        _fail("metadata JSON top level is not an object")
    if _canonical_json_bytes_v1(raw) != payload:
        _fail("metadata JSON is not canonical ASCII JSON")
    return raw


def _closed_dict_v1(
    value: object,
    expected_fields: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or tuple(sorted(value)) != tuple(sorted(expected_fields)):
        _fail(f"{label} closed field manifest differs")
    return value


def _exact_text_v1(
    value: object,
    label: str,
    *,
    maximum: int = _MAXIMUM_ID_BYTES_V1,
    ascii_only: bool = True,
) -> str:
    if type(value) is not str or not value or "\x00" in value:
        _fail(f"{label} is not exact nonempty text")
    try:
        encoded = value.encode("ascii" if ascii_only else "utf-8")
    except UnicodeError as exc:
        _fail(f"{label} encoding differs: {exc}")
    if len(encoded) > maximum:
        _fail(f"{label} exceeds its byte bound")
    return value


def _sha256_v1(value: object, label: str) -> str:
    text = _exact_text_v1(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        _fail(f"{label} is not lowercase SHA-256")
    return text


def _content_address_v1(value: object, label: str, *, prefix: str | None = None) -> str:
    text = _exact_text_v1(value, label)
    match = re.fullmatch(r"([a-z][a-z0-9_]*_)([0-9a-f]{64})", text)
    if match is None or (prefix is not None and match.group(1) != prefix):
        _fail(f"{label} content-address shape differs")
    return text


def _content_id_v1(mapping: Mapping[str, object], *, domain: bytes, prefix: str) -> str:
    return prefix + hashlib.sha256(domain + _canonical_json_bytes_v1(dict(mapping))).hexdigest()


def _run_manifest_id_v1(mapping: Mapping[str, str]) -> str:
    if tuple(mapping) != _RUN_FIELDS_WITHOUT_ID_V1:
        _fail("run manifest preimage declared order differs")
    preimage = [[name, mapping[name]] for name in _RUN_FIELDS_WITHOUT_ID_V1]
    return _RUN_MANIFEST_ID_PREFIX_V1 + hashlib.sha256(
        _RUN_MANIFEST_DOMAIN_V1 + _canonical_json_bytes_v1(preimage)
    ).hexdigest()


def _exact_int_v1(value: object, label: str, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        _fail(f"{label} exact integer bound differs")
    return value


def _claims_v1(value: object) -> dict[str, bool]:
    raw = _closed_dict_v1(value, _FALSE_CLAIMS_V1, "bundle claims")
    result: dict[str, bool] = {}
    for name in _FALSE_CLAIMS_V1:
        item = raw[name]
        if type(item) is not bool or item is not False:
            _fail(f"bundle claim {name} must remain exact false")
        result[name] = False
    return result


def _section_manifest_v1(value: object, label: str) -> tuple[int, str]:
    raw = _closed_dict_v1(value, _SECTION_FIELDS_V1, label)
    return (
        _exact_int_v1(raw["length"], f"{label} length", maximum=MAXIMUM_PHASE2B_SCORER_BUNDLE_BYTES_V1),
        _sha256_v1(raw["sha256"], f"{label} SHA-256"),
    )


def _opening_v1(value: object, label: str) -> tuple[str, str, str]:
    raw = _closed_dict_v1(value, _OPENING_FIELDS_V1, label)
    manifest_sha = _sha256_v1(raw["manifest_sha256"], f"{label} manifest SHA")
    salt = _exact_text_v1(
        raw["salt"],
        f"{label} salt",
        maximum=_MAXIMUM_SALT_BYTES_V1,
        ascii_only=False,
    )
    if len(salt.encode("utf-8")) < 32:
        _fail(f"{label} salt is shorter than 32 bytes")
    commitment = _sha256_v1(raw["commitment_sha256"], f"{label} commitment")
    return manifest_sha, salt, commitment


def _exact_string_list_v1(
    value: object,
    label: str,
    *,
    count: int | None = None,
    maximum_item_bytes: int = _MAXIMUM_ID_BYTES_V1,
) -> tuple[str, ...]:
    if type(value) is not list or (count is not None and len(value) != count):
        _fail(f"{label} list count differs")
    result = tuple(
        _exact_text_v1(item, f"{label} item", maximum=maximum_item_bytes)
        for item in value
    )
    if len(set(result)) != len(result):
        _fail(f"{label} contains duplicates")
    return result


def _answer_row_mapping_v1(value: FormalUnsealedAnswerRowV2) -> dict[str, object]:
    if type(value) is not FormalUnsealedAnswerRowV2:
        _fail("answer row exact type differs")
    family = value.canonical_family_id
    if family is not None and type(family) is not CanonicalFamilyId:
        _fail("answer row family exact enum differs")
    if type(value.case_type) is not Phase2BCaseType or type(value.expected_decision) is not PredictionDecisionV2:
        _fail("answer row case or decision exact enum differs")
    if type(value.binding) is not tuple or type(value.admissible_scale_ids) is not tuple:
        _fail("answer row tuple field type differs")
    binding: list[dict[str, str]] = []
    for item in value.binding:
        if type(item) is not RoleBinding:
            _fail("answer row binding exact type differs")
        binding.append({"entity_id": item.entity_id, "role_id": item.role_id})
    return {
        "admissible_scale_ids": list(value.admissible_scale_ids),
        "binding": binding,
        "canonical_family_id": None if family is None else family.value,
        "case_type": value.case_type.value,
        "expected_decision": value.expected_decision.value,
        "input_row_id": value.input_row_id,
    }


def _decode_answer_rows_v1(value: object) -> tuple[FormalUnsealedAnswerRowV2, ...]:
    if type(value) is not list or len(value) != _EXACT_MAIN_COUNT_V1:
        _fail("answer row count differs from 720")
    rows: list[FormalUnsealedAnswerRowV2] = []
    for index, item in enumerate(value):
        raw = _closed_dict_v1(item, _ANSWER_ROW_FIELDS_V1, f"answer row {index}")
        raw_binding = raw["binding"]
        if type(raw_binding) is not list or len(raw_binding) > 64:
            _fail(f"answer row {index} binding count differs")
        binding: list[RoleBinding] = []
        for binding_index, entry in enumerate(raw_binding):
            closed = _closed_dict_v1(
                entry,
                ("entity_id", "role_id"),
                f"answer row {index} binding {binding_index}",
            )
            binding.append(
                RoleBinding(
                    role_id=_exact_text_v1(closed["role_id"], "answer role ID"),
                    entity_id=_exact_text_v1(closed["entity_id"], "answer entity ID"),
                )
            )
        scales = _exact_string_list_v1(
            raw["admissible_scale_ids"],
            f"answer row {index} scales",
        )
        family_raw = raw["canonical_family_id"]
        family = None if family_raw is None else CanonicalFamilyId(
            _exact_text_v1(family_raw, f"answer row {index} family")
        )
        rows.append(
            FormalUnsealedAnswerRowV2(
                input_row_id=_content_address_v1(
                    raw["input_row_id"],
                    f"answer row {index} input ID",
                    prefix="phase2b_recognizer_input_row_v2_",
                ),
                case_type=Phase2BCaseType(
                    _exact_text_v1(raw["case_type"], f"answer row {index} case")
                ),
                expected_decision=PredictionDecisionV2(
                    _exact_text_v1(raw["expected_decision"], f"answer row {index} decision")
                ),
                canonical_family_id=family,
                binding=tuple(binding),
                admissible_scale_ids=scales,
            )
        )
    return tuple(rows)


def _gate_row_mapping_v1(value: FormalUnsealedGateInputRowV2) -> dict[str, object]:
    if type(value) is not FormalUnsealedGateInputRowV2:
        _fail("gate row exact type differs")
    if (
        type(value.case_type) is not Phase2BCaseType
        or type(value.margin_stratum) is not MarginStratum
        or type(value.canonical_family_id) is not CanonicalFamilyId
        or type(value.scale_slice_id) is not FormalUnsealedScaleSliceIdV2
    ):
        _fail("gate row enum exact type differs")
    return {
        "canonical_family_id": value.canonical_family_id.value,
        "case_type": value.case_type.value,
        "input_row_id": value.input_row_id,
        "latent_base_case_id": value.latent_base_case_id,
        "margin_stratum": value.margin_stratum.value,
        "scale_slice_id": value.scale_slice_id.value,
    }


@dataclass(frozen=True, slots=True)
class _ParsedGateRowV1:
    input_row_id: str
    case_type: Phase2BCaseType
    margin_stratum: MarginStratum
    canonical_family_id: CanonicalFamilyId
    scale_slice_id: FormalUnsealedScaleSliceIdV2
    latent_base_case_id: str


def _decode_gate_rows_v1(value: object) -> tuple[_ParsedGateRowV1, ...]:
    if type(value) is not list or len(value) != _EXACT_MAIN_COUNT_V1:
        _fail("gate row count differs from 720")
    rows: list[_ParsedGateRowV1] = []
    for index, item in enumerate(value):
        raw = _closed_dict_v1(item, _GATE_ROW_FIELDS_V1, f"gate row {index}")
        rows.append(
            _ParsedGateRowV1(
                input_row_id=_content_address_v1(raw["input_row_id"], "gate input row ID", prefix="phase2b_recognizer_input_row_v2_"),
                case_type=Phase2BCaseType(_exact_text_v1(raw["case_type"], "gate case type")),
                margin_stratum=MarginStratum(_exact_text_v1(raw["margin_stratum"], "gate margin stratum")),
                canonical_family_id=CanonicalFamilyId(_exact_text_v1(raw["canonical_family_id"], "gate family")),
                scale_slice_id=FormalUnsealedScaleSliceIdV2(_exact_text_v1(raw["scale_slice_id"], "gate scale slice")),
                latent_base_case_id=_content_address_v1(raw["latent_base_case_id"], "gate latent ID", prefix="phase2b_latent_base_case_v2_"),
            )
        )
    input_ids = tuple(item.input_row_id for item in rows)
    if len(set(input_ids)) != _EXACT_MAIN_COUNT_V1:
        _fail("gate input row IDs are not unique")
    return tuple(rows)


def _materialize_gate_rows_v1(
    parsed: tuple[_ParsedGateRowV1, ...],
    answer: FormalUnsealedAnswerManifestV2,
) -> tuple[FormalUnsealedGateInputRowV2, ...]:
    by_input = {item.input_row_id: item.answer_row_id for item in answer.main_answer_rows}
    if len(by_input) != _EXACT_MAIN_COUNT_V1:
        _fail("rebuilt answer rows are not an exact input-row bijection")
    if set(by_input) != {item.input_row_id for item in parsed}:
        _fail("gate rows do not join exactly to rebuilt answer rows")
    return tuple(
        FormalUnsealedGateInputRowV2(
            input_row_id=item.input_row_id,
            answer_row_id=by_input[item.input_row_id],
            case_type=item.case_type,
            margin_stratum=item.margin_stratum,
            canonical_family_id=item.canonical_family_id,
            scale_slice_id=item.scale_slice_id,
            latent_base_case_id=item.latent_base_case_id,
        )
        for item in parsed
    )


def _report_primitive_v1(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        return {"binary64_hex": value.hex()}
    if isinstance(value, Enum):
        return _report_primitive_v1(value.value)
    if type(value) is tuple:
        return [_report_primitive_v1(item) for item in value]
    if type(value) is list:
        return [_report_primitive_v1(item) for item in value]
    if type(value) is dict:
        return {
            _exact_text_v1(key, "report mapping key"): _report_primitive_v1(item)
            for key, item in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _report_primitive_v1(object.__getattribute__(value, item.name))
            for item in fields(value)
        }
    _fail(f"report contains unsupported exact type {type(value).__name__}")


def _deep_freeze_v1(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({key: _deep_freeze_v1(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_deep_freeze_v1(item) for item in value)
    return value


@dataclass(frozen=True, slots=True, init=False)
class DecodedPhase2BScorerBundleV1:
    bundle_id: str
    bundle_sha256: str
    bundle_bytes: bytes
    metadata: Mapping[str, object]
    execution_freeze_bytes: bytes
    input_archive_bytes: bytes
    prediction_archive_bytes: bytes
    execution_freeze_manifest: ExecutionFreezeManifest
    decoded_input_archive: DecodedRecognizerInputArchiveV2
    decoded_prediction_archive: DecodedRecognizerPredictionArchiveV2
    structural_receipt: StrictRecognizerStructuralReceiptV2
    partition_manifest: UnsealedPredictionPartitionManifestV2
    structural_evaluation: UnsealedPredictionStructuralEvaluationV2
    answer_manifest: FormalUnsealedAnswerManifestV2
    gate_input_manifest: FormalUnsealedGateInputManifestV2
    authority_requirement_id: str
    actor_profile_content_id: str
    actor_profile_canonical_sha256: str
    campaign_id: str
    attempt_id: str
    execution_registry_id: str
    scorer_image_digest: str
    scorer_source_root_sha256: str
    protocol_id: str
    exact_freeze_id: str
    execution_freeze_manifest_id: str
    scoring_source_sha256: str
    run_manifest_id: str
    prediction_if_absent_receipt_id: str
    signed_reveal_token_id: str
    answer_commitment_salt: str
    answer_commitment_sha256: str
    gate_input_commitment_salt: str
    gate_input_commitment_sha256: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("scorer bundle DTOs are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class EvaluatedPhase2BScorerBundleV1:
    scoring_mechanics: Unsealed960PredictionScoringMechanicsV2
    replay_input_contract: ActualUnsealed960ReplayInputContractV2
    available_gate_mechanics: Unsealed960AvailableGateMechanicsV2
    score_report_bytes: bytes
    score_report_id: str
    available_gate_report_bytes: bytes
    available_gate_report_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("scorer evaluation DTOs are privately issued")


# Runtime-facing exact class alias.
Phase2BScorerBundleEvaluationV1 = EvaluatedPhase2BScorerBundleV1

_DECODED_FIELDS_V1: Final = (
    "bundle_id",
    "bundle_sha256",
    "bundle_bytes",
    "metadata",
    "execution_freeze_bytes",
    "input_archive_bytes",
    "prediction_archive_bytes",
    "execution_freeze_manifest",
    "decoded_input_archive",
    "decoded_prediction_archive",
    "structural_receipt",
    "partition_manifest",
    "structural_evaluation",
    "answer_manifest",
    "gate_input_manifest",
    "authority_requirement_id",
    "actor_profile_content_id",
    "actor_profile_canonical_sha256",
    "campaign_id",
    "attempt_id",
    "execution_registry_id",
    "scorer_image_digest",
    "scorer_source_root_sha256",
    "protocol_id",
    "exact_freeze_id",
    "execution_freeze_manifest_id",
    "scoring_source_sha256",
    "run_manifest_id",
    "prediction_if_absent_receipt_id",
    "signed_reveal_token_id",
    "answer_commitment_salt",
    "answer_commitment_sha256",
    "gate_input_commitment_salt",
    "gate_input_commitment_sha256",
)
_EVALUATED_FIELDS_V1: Final = (
    "scoring_mechanics",
    "replay_input_contract",
    "available_gate_mechanics",
    "score_report_bytes",
    "score_report_id",
    "available_gate_report_bytes",
    "available_gate_report_id",
)
if tuple(item.name for item in fields(DecodedPhase2BScorerBundleV1)) != _DECODED_FIELDS_V1:
    raise RuntimeError("decoded scorer bundle field manifest drift")
if tuple(item.name for item in fields(EvaluatedPhase2BScorerBundleV1)) != _EVALUATED_FIELDS_V1:
    raise RuntimeError("evaluated scorer bundle field manifest drift")


def _parse_run_manifest_v1(value: object) -> dict[str, str]:
    raw = _closed_dict_v1(value, _RUN_FIELDS_V1, "scorer run manifest")
    result: dict[str, str] = {}
    for name in _RUN_FIELDS_WITHOUT_ID_V1:
        item = raw[name]
        if name.endswith("_sha256"):
            result[name] = _sha256_v1(item, f"run manifest {name}")
        elif name == "authority_requirement_id":
            result[name] = _content_address_v1(
                item,
                f"run manifest {name}",
                prefix=(
                    "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
                ),
            )
        elif name == "actor_profile_content_id":
            result[name] = _content_address_v1(
                item,
                f"run manifest {name}",
                prefix="phase2b_c1_container_actor_profile_v2_",
            )
        elif name == "campaign_id":
            result[name] = _content_address_v1(
                item,
                f"run manifest {name}",
                prefix="phase2b_actual_unsealed_960_campaign_v2_",
            )
        elif name == "attempt_id":
            result[name] = _content_address_v1(
                item,
                f"run manifest {name}",
                prefix="phase2b_actual_unsealed_960_attempt_v2_",
            )
        elif name == "execution_registry_id":
            result[name] = _content_address_v1(
                item,
                f"run manifest {name}",
                prefix="phase2b_c1_owner_accepted_execution_registry_v1_",
            )
        elif name == "scorer_image_digest":
            text = _exact_text_v1(item, f"run manifest {name}")
            if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
                _fail("run manifest scorer image digest shape differs")
            result[name] = text
        elif name == "protocol_id":
            result[name] = _content_address_v1(item, f"run manifest {name}", prefix="phase2b_protocol_")
        elif name == "exact_freeze_id":
            result[name] = _content_address_v1(item, f"run manifest {name}", prefix="phase2b_exact_freeze_")
        elif name == "execution_freeze_manifest_id":
            result[name] = _content_address_v1(item, f"run manifest {name}", prefix="phase2b_execution_freeze_")
        elif name == "input_archive_id":
            result[name] = _content_address_v1(item, f"run manifest {name}", prefix="phase2b_recognizer_input_archive_v2_")
        elif name == "prediction_archive_id":
            result[name] = _content_address_v1(item, f"run manifest {name}", prefix="phase2b_recognizer_prediction_archive_v2_")
        elif name == "prediction_if_absent_receipt_id":
            result[name] = _content_address_v1(
                item,
                f"run manifest {name}",
                prefix="phase2b_prediction_if_absent_receipt_v1_",
            )
        elif name == "signed_reveal_token_id":
            result[name] = _content_address_v1(
                item,
                f"run manifest {name}",
                prefix="phase2b_signed_reveal_token_v1_",
            )
        else:
            result[name] = _content_address_v1(item, f"run manifest {name}")
    supplied_id = _content_address_v1(
        raw["run_manifest_id"],
        "scorer run manifest ID",
        prefix=_RUN_MANIFEST_ID_PREFIX_V1,
    )
    expected_id = _run_manifest_id_v1(result)
    if supplied_id != expected_id:
        _fail("scorer run manifest content address differs")
    result["run_manifest_id"] = supplied_id
    return result


def encode_phase2b_scorer_bundle_v1(
    *,
    authority_requirement_id: str,
    actor_profile_content_id: str,
    actor_profile_canonical_sha256: str,
    campaign_id: str,
    attempt_id: str,
    execution_registry_id: str,
    scorer_image_digest: str,
    scorer_source_root_sha256: str,
    execution_freeze_bytes: bytes,
    input_archive: bytes,
    prediction_archive: bytes,
    main_row_ids: tuple[str, ...],
    semantic_conflict_row_ids: tuple[str, ...],
    answer_manifest: FormalUnsealedAnswerManifestV2,
    answer_commitment_salt: str,
    salted_answer_commitment_sha256: str,
    gate_input_manifest: FormalUnsealedGateInputManifestV2,
    gate_input_commitment_salt: str,
    salted_gate_input_commitment_sha256: str,
    prediction_if_absent_receipt_id: str,
    signed_reveal_token_id: str,
) -> bytes:
    """Encode one exact closed bundle and self-decode it before returning."""

    if (
        type(execution_freeze_bytes) is not bytes
        or type(input_archive) is not bytes
        or type(prediction_archive) is not bytes
        or type(main_row_ids) is not tuple
        or type(semantic_conflict_row_ids) is not tuple
        or type(answer_manifest) is not FormalUnsealedAnswerManifestV2
        or type(gate_input_manifest) is not FormalUnsealedGateInputManifestV2
    ):
        _fail("bundle encoder exact top-level type differs")
    actor_profile = _content_address_v1(
        actor_profile_content_id,
        "actor profile content ID",
        prefix="phase2b_c1_container_actor_profile_v2_",
    )
    if actor_profile != PHASE2B_SCORER_ACTOR_PROFILE_CONTENT_ID_V1:
        _fail("actor profile content ID differs from the frozen V2 profile")
    authority_requirement = _content_address_v1(
        authority_requirement_id,
        "authority requirement ID",
        prefix="phase2b_c1_owner_accepted_container_authority_requirement_v1_",
    )
    if authority_requirement != PHASE2B_C1_OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V1:
        _fail("authority requirement ID differs from the frozen owner acceptance")
    profile_sha = _sha256_v1(
        actor_profile_canonical_sha256,
        "actor profile canonical SHA",
    )
    if profile_sha != PHASE2B_SCORER_ACTOR_PROFILE_CANONICAL_SHA256_V1:
        _fail("actor profile canonical SHA differs from the frozen V2 profile")
    campaign = _content_address_v1(
        campaign_id,
        "campaign ID",
        prefix="phase2b_actual_unsealed_960_campaign_v2_",
    )
    attempt = _content_address_v1(
        attempt_id,
        "attempt ID",
        prefix="phase2b_actual_unsealed_960_attempt_v2_",
    )
    registry = _content_address_v1(
        execution_registry_id,
        "execution registry ID",
        prefix="phase2b_c1_owner_accepted_execution_registry_v1_",
    )
    image_digest = _exact_text_v1(scorer_image_digest, "scorer image digest")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest) is None:
        _fail("scorer image digest shape differs")
    source_root_sha = _sha256_v1(
        scorer_source_root_sha256,
        "scorer source-root SHA",
    )
    receipt_id = _content_address_v1(
        prediction_if_absent_receipt_id,
        "prediction IF_ABSENT receipt ID",
        prefix="phase2b_prediction_if_absent_receipt_v1_",
    )
    reveal_id = _content_address_v1(
        signed_reveal_token_id,
        "signed reveal token ID",
        prefix="phase2b_signed_reveal_token_v1_",
    )
    if receipt_id == reveal_id:
        _fail("prediction receipt and reveal token identities must be distinct")
    answer_salt = _exact_text_v1(
        answer_commitment_salt,
        "answer commitment salt",
        maximum=_MAXIMUM_SALT_BYTES_V1,
        ascii_only=False,
    )
    gate_salt = _exact_text_v1(
        gate_input_commitment_salt,
        "gate commitment salt",
        maximum=_MAXIMUM_SALT_BYTES_V1,
        ascii_only=False,
    )
    answer_commitment = _sha256_v1(
        salted_answer_commitment_sha256,
        "answer commitment SHA",
    )
    gate_commitment = _sha256_v1(
        salted_gate_input_commitment_sha256,
        "gate commitment SHA",
    )
    if len(main_row_ids) != _EXACT_MAIN_COUNT_V1 or len(semantic_conflict_row_ids) != _EXACT_CHALLENGE_COUNT_V1:
        _fail("bundle partition must be exact 720/240")
    if any(type(item) is not str for item in (*main_row_ids, *semantic_conflict_row_ids)):
        _fail("bundle partition IDs require exact strings")
    raw_answer_rows = object.__getattribute__(answer_manifest, "main_answer_rows")
    raw_gate_rows = object.__getattribute__(gate_input_manifest, "main_gate_input_rows")
    if type(raw_answer_rows) is not tuple or len(raw_answer_rows) != _EXACT_MAIN_COUNT_V1:
        _fail("answer manifest does not contain exact 720 rows")
    if type(raw_gate_rows) is not tuple or len(raw_gate_rows) != _EXACT_MAIN_COUNT_V1:
        _fail("gate manifest does not contain exact 720 rows")

    # Public strict verification closes all raw bytes before any bundle hash.
    receipt = verify_strict_recognizer_io_structure_v2(
        input_archive=input_archive,
        prediction_archive=prediction_archive,
    )
    freeze = decode_execution_freeze_manifest_json_v2(execution_freeze_bytes)
    if type(receipt) is not StrictRecognizerStructuralReceiptV2:
        _fail("strict verifier returned a nonexact receipt")
    if type(freeze) is not ExecutionFreezeManifest:
        _fail("execution-freeze decoder returned a nonexact manifest")
    if freeze.manifest_id != receipt.execution_freeze_manifest_id:
        _fail("execution-freeze raw section does not bind prediction context")
    decoded_prediction = decode_public_recognizer_prediction_archive_v2(
        prediction_archive
    )
    if type(decoded_prediction) is not DecodedRecognizerPredictionArchiveV2:
        _fail("prediction decoder returned a nonexact archive")
    partition = build_unsealed_prediction_partition_manifest_v2(
        prediction_archive=decoded_prediction,
        main_row_ids=main_row_ids,
        semantic_conflict_row_ids=semantic_conflict_row_ids,
    )
    if type(partition) is not UnsealedPredictionPartitionManifestV2:
        _fail("partition builder returned a nonexact manifest")
    exact_freeze_id = partition.exact_freeze_id
    run_without_id: dict[str, str] = {
        "authority_requirement_id": authority_requirement,
        "actor_profile_content_id": actor_profile,
        "actor_profile_canonical_sha256": profile_sha,
        "campaign_id": campaign,
        "attempt_id": attempt,
        "execution_registry_id": registry,
        "scorer_image_digest": image_digest,
        "scorer_source_root_sha256": source_root_sha,
        "protocol_id": receipt.protocol_id,
        "exact_freeze_id": exact_freeze_id,
        "execution_freeze_manifest_id": freeze.manifest_id,
        "input_archive_id": receipt.input_archive_id,
        "input_archive_sha256": hashlib.sha256(input_archive).hexdigest(),
        "prediction_archive_id": receipt.prediction_archive_id,
        "prediction_archive_sha256": hashlib.sha256(prediction_archive).hexdigest(),
        "answer_commitment_sha256": answer_commitment,
        "gate_input_commitment_sha256": gate_commitment,
        "prediction_if_absent_receipt_id": receipt_id,
        "signed_reveal_token_id": reveal_id,
        "scoring_source_sha256": PHASE2B_SCORING_SOURCE_SHA256_V1,
    }
    run_manifest = dict(run_without_id)
    run_manifest["run_manifest_id"] = _run_manifest_id_v1(run_without_id)
    sections = {
        "execution_freeze": {
            "length": len(execution_freeze_bytes),
            "sha256": hashlib.sha256(execution_freeze_bytes).hexdigest(),
        },
        "input_archive": {
            "length": len(input_archive),
            "sha256": hashlib.sha256(input_archive).hexdigest(),
        },
        "prediction_archive": {
            "length": len(prediction_archive),
            "sha256": hashlib.sha256(prediction_archive).hexdigest(),
        },
    }
    metadata = {
        "answer_opening": {
            "commitment_sha256": answer_commitment,
            "manifest_sha256": answer_manifest.answer_manifest_sha256,
            "salt": answer_salt,
        },
        "answer_rows": [_answer_row_mapping_v1(item) for item in raw_answer_rows],
        "claims": {name: False for name in _FALSE_CLAIMS_V1},
        "gate_opening": {
            "commitment_sha256": gate_commitment,
            "manifest_sha256": gate_input_manifest.gate_input_manifest_sha256,
            "salt": gate_salt,
        },
        "gate_rows": [_gate_row_mapping_v1(item) for item in raw_gate_rows],
        "main_row_ids": list(main_row_ids),
        "run_manifest": run_manifest,
        "schema_version": PHASE2B_SCORER_BUNDLE_V1_VERSION,
        "sections": sections,
        "semantic_conflict_row_ids": list(semantic_conflict_row_ids),
    }
    metadata_bytes = _canonical_json_bytes_v1(metadata)
    if not 1 <= len(metadata_bytes) <= MAXIMUM_PHASE2B_SCORER_METADATA_BYTES_V1:
        _fail("encoded scorer metadata exceeds its byte cap")
    header = _HEADER_V1.pack(
        PHASE2B_SCORER_BUNDLE_V1_MAGIC,
        len(metadata_bytes),
        len(execution_freeze_bytes),
        len(input_archive),
        len(prediction_archive),
    )
    bundle = b"".join(
        (header, metadata_bytes, execution_freeze_bytes, input_archive, prediction_archive)
    )
    if len(bundle) > MAXIMUM_PHASE2B_SCORER_BUNDLE_BYTES_V1:
        _fail("encoded scorer bundle exceeds its byte cap")
    # The exact decoder is the final authority for encoder parity.
    decode_phase2b_scorer_bundle_v1(bundle)
    return bundle


def _decode_phase2b_scorer_bundle_v1_impl(
    bundle: bytes,
) -> DecodedPhase2BScorerBundleV1:
    """Decode and reconstruct all private upstream DTOs through public APIs."""

    if type(bundle) is not bytes or not (
        _HEADER_V1.size < len(bundle) <= MAXIMUM_PHASE2B_SCORER_BUNDLE_BYTES_V1
    ):
        _fail("scorer bundle exact byte bound differs")
    try:
        magic, metadata_length, freeze_length, input_length, prediction_length = _HEADER_V1.unpack_from(bundle, 0)
    except struct.error as exc:
        _fail(f"scorer bundle header is truncated: {exc}")
    if magic != PHASE2B_SCORER_BUNDLE_V1_MAGIC:
        _fail("scorer bundle magic differs")
    lengths = (metadata_length, freeze_length, input_length, prediction_length)
    if any(type(item) is not int for item in lengths):
        _fail("scorer bundle header lengths have wrong type")
    if not 1 <= metadata_length <= MAXIMUM_PHASE2B_SCORER_METADATA_BYTES_V1:
        _fail("scorer bundle metadata length differs")
    if not 1 <= freeze_length <= MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2:
        _fail("scorer bundle freeze length differs")
    if not ARCHIVE_HEADER_BYTES_V2 <= input_length <= MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2:
        _fail("scorer bundle input archive length differs")
    if not PREDICTION_ARCHIVE_HEADER_BYTES_V2 <= prediction_length <= MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2:
        _fail("scorer bundle prediction archive length differs")
    total = _HEADER_V1.size + sum(lengths)
    if total != len(bundle):
        _fail("scorer bundle total length or trailing bytes differ")
    cursor = _HEADER_V1.size
    metadata_bytes = bundle[cursor : cursor + metadata_length]
    cursor += metadata_length
    freeze_bytes = bundle[cursor : cursor + freeze_length]
    cursor += freeze_length
    input_bytes = bundle[cursor : cursor + input_length]
    cursor += input_length
    prediction_bytes = bundle[cursor : cursor + prediction_length]

    metadata = _decode_canonical_json_v1(metadata_bytes)
    _closed_dict_v1(metadata, _TOP_LEVEL_METADATA_FIELDS_V1, "bundle metadata")
    if _exact_text_v1(metadata["schema_version"], "bundle schema version") != PHASE2B_SCORER_BUNDLE_V1_VERSION:
        _fail("bundle schema version differs")
    _claims_v1(metadata["claims"])
    raw_sections = _closed_dict_v1(metadata["sections"], _SECTION_NAMES_V1, "bundle sections")
    section_values = {
        "execution_freeze": freeze_bytes,
        "input_archive": input_bytes,
        "prediction_archive": prediction_bytes,
    }
    for name in _SECTION_NAMES_V1:
        declared_length, declared_sha = _section_manifest_v1(raw_sections[name], f"section {name}")
        raw = section_values[name]
        if declared_length != len(raw) or declared_sha != hashlib.sha256(raw).hexdigest():
            _fail(f"section {name} byte binding differs")

    main_ids = _exact_string_list_v1(metadata["main_row_ids"], "main partition", count=_EXACT_MAIN_COUNT_V1)
    challenge_ids = _exact_string_list_v1(
        metadata["semantic_conflict_row_ids"],
        "semantic-conflict partition",
        count=_EXACT_CHALLENGE_COUNT_V1,
    )
    if set(main_ids).intersection(challenge_ids):
        _fail("720/240 partition overlaps")
    answer_rows = _decode_answer_rows_v1(metadata["answer_rows"])
    parsed_gate_rows = _decode_gate_rows_v1(metadata["gate_rows"])
    answer_sha, answer_salt, answer_commitment = _opening_v1(metadata["answer_opening"], "answer opening")
    gate_sha, gate_salt, gate_commitment = _opening_v1(metadata["gate_opening"], "gate opening")
    run = _parse_run_manifest_v1(metadata["run_manifest"])
    if run["scoring_source_sha256"] != PHASE2B_SCORING_SOURCE_SHA256_V1:
        _fail("run manifest scoring source SHA differs")
    if (
        run["authority_requirement_id"]
        != PHASE2B_C1_OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V1
    ):
        _fail("run manifest authority requirement differs")
    if run["actor_profile_content_id"] != PHASE2B_SCORER_ACTOR_PROFILE_CONTENT_ID_V1:
        _fail("run manifest actor profile differs from the frozen V2 identity")
    if (
        run["actor_profile_canonical_sha256"]
        != PHASE2B_SCORER_ACTOR_PROFILE_CANONICAL_SHA256_V1
    ):
        _fail("run manifest actor profile canonical SHA differs")
    if run["answer_commitment_sha256"] != answer_commitment or run["gate_input_commitment_sha256"] != gate_commitment:
        _fail("run manifest commitment cross-binding differs")
    if run["prediction_if_absent_receipt_id"] == run["signed_reveal_token_id"]:
        _fail("run manifest freshness identities are not distinct")

    # Each public primitive below is deliberately called independently.
    freeze = decode_execution_freeze_manifest_json_v2(freeze_bytes)
    decoded_input = decode_public_recognizer_input_archive_v2(input_bytes)
    decoded_prediction = decode_public_recognizer_prediction_archive_v2(prediction_bytes)
    receipt = verify_strict_recognizer_io_structure_v2(
        input_archive=input_bytes,
        prediction_archive=prediction_bytes,
    )
    if type(freeze) is not ExecutionFreezeManifest:
        _fail("execution-freeze decoder returned a nonexact manifest")
    if type(decoded_input) is not DecodedRecognizerInputArchiveV2:
        _fail("input decoder returned a nonexact archive")
    if type(decoded_prediction) is not DecodedRecognizerPredictionArchiveV2:
        _fail("prediction decoder returned a nonexact archive")
    if type(receipt) is not StrictRecognizerStructuralReceiptV2:
        _fail("strict verifier returned a nonexact receipt")
    partition = build_unsealed_prediction_partition_manifest_v2(
        prediction_archive=decoded_prediction,
        main_row_ids=main_ids,
        semantic_conflict_row_ids=challenge_ids,
    )
    if type(partition) is not UnsealedPredictionPartitionManifestV2:
        _fail("partition builder returned a nonexact manifest")
    evaluation_candidate = evaluate_unsealed_prediction_archive_structure_v2(
        prediction_archive=decoded_prediction,
        partition_manifest=partition,
    )
    if type(evaluation_candidate) is not UnsealedPredictionStructuralEvaluationV2:
        if type(evaluation_candidate) is UnsealedPredictionEvaluationRejectionV2:
            _fail(f"structural evaluation rejected: {evaluation_candidate.reason}")
        _fail("structural evaluator returned a nonexact type")
    evaluation = evaluation_candidate
    answer = build_formal_unsealed_answer_manifest_v2(
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
        main_answer_rows=answer_rows,
    )
    if answer.answer_manifest_sha256 != answer_sha:
        _fail("rebuilt answer manifest differs")
    formal_candidate = validate_formal_unsealed_prediction_scoring_contract_v2(
        structural_receipt=receipt,
        structural_evaluation=evaluation,
        partition_manifest=partition,
        answer_manifest=answer,
        revealed_answer_manifest_sha256=answer_sha,
        answer_commitment_salt=answer_salt,
        salted_answer_commitment_sha256=answer_commitment,
    )
    if type(formal_candidate) is not FormalUnsealedPredictionScoringContractValidationV2:
        if type(formal_candidate) is FormalUnsealedPredictionScoringContractRejectionV2:
            _fail(f"answer opening rejected: {formal_candidate.reason.value}")
        _fail("answer opening validator returned a nonexact type")
    gate_rows = _materialize_gate_rows_v1(parsed_gate_rows, answer)
    gate_manifest = build_formal_unsealed_gate_input_manifest_v2(
        answer_manifest=answer,
        main_gate_input_rows=gate_rows,
    )
    if gate_manifest.gate_input_manifest_sha256 != gate_sha:
        _fail("rebuilt gate manifest differs")
    if (
        salted_gate_input_commitment_sha256_v2(gate_sha, gate_salt)
        != gate_commitment
    ):
        _fail("gate commitment opening differs")

    exact_groups = (
        (freeze.manifest_id, receipt.execution_freeze_manifest_id, run["execution_freeze_manifest_id"]),
        (receipt.protocol_id, answer.phase2b_protocol_id, run["protocol_id"]),
        (partition.exact_freeze_id, answer.exact_freeze_id, run["exact_freeze_id"]),
        (receipt.input_archive_id, decoded_input.archive_id, run["input_archive_id"]),
        (receipt.prediction_archive_id, decoded_prediction.archive_id, run["prediction_archive_id"]),
        (receipt.input_archive_sha256, hashlib.sha256(input_bytes).hexdigest(), run["input_archive_sha256"]),
        (receipt.prediction_archive_sha256, hashlib.sha256(prediction_bytes).hexdigest(), run["prediction_archive_sha256"]),
    )
    if any(any(type(item) is not str or item != group[0] for item in group[1:]) for group in exact_groups):
        _fail("bundle run-manifest graph cross-binding differs")

    bundle_sha = hashlib.sha256(bundle).hexdigest()
    value = object.__new__(DecodedPhase2BScorerBundleV1)
    issued = (
        ("bundle_id", _BUNDLE_ID_PREFIX_V1 + bundle_sha),
        ("bundle_sha256", bundle_sha),
        ("bundle_bytes", bytes(bundle)),
        ("metadata", _deep_freeze_v1(metadata)),
        ("execution_freeze_bytes", bytes(freeze_bytes)),
        ("input_archive_bytes", bytes(input_bytes)),
        ("prediction_archive_bytes", bytes(prediction_bytes)),
        ("execution_freeze_manifest", freeze),
        ("decoded_input_archive", decoded_input),
        ("decoded_prediction_archive", decoded_prediction),
        ("structural_receipt", receipt),
        ("partition_manifest", partition),
        ("structural_evaluation", evaluation),
        ("answer_manifest", answer),
        ("gate_input_manifest", gate_manifest),
        ("authority_requirement_id", run["authority_requirement_id"]),
        ("actor_profile_content_id", run["actor_profile_content_id"]),
        ("actor_profile_canonical_sha256", run["actor_profile_canonical_sha256"]),
        ("campaign_id", run["campaign_id"]),
        ("attempt_id", run["attempt_id"]),
        ("execution_registry_id", run["execution_registry_id"]),
        ("scorer_image_digest", run["scorer_image_digest"]),
        ("scorer_source_root_sha256", run["scorer_source_root_sha256"]),
        ("protocol_id", run["protocol_id"]),
        ("exact_freeze_id", run["exact_freeze_id"]),
        ("execution_freeze_manifest_id", run["execution_freeze_manifest_id"]),
        ("scoring_source_sha256", run["scoring_source_sha256"]),
        ("run_manifest_id", run["run_manifest_id"]),
        ("prediction_if_absent_receipt_id", run["prediction_if_absent_receipt_id"]),
        ("signed_reveal_token_id", run["signed_reveal_token_id"]),
        ("answer_commitment_salt", answer_salt),
        ("answer_commitment_sha256", answer_commitment),
        ("gate_input_commitment_salt", gate_salt),
        ("gate_input_commitment_sha256", gate_commitment),
    )
    for name, item in issued:
        object.__setattr__(value, name, item)
    return value


def decode_phase2b_scorer_bundle_v1(
    bundle: bytes,
) -> DecodedPhase2BScorerBundleV1:
    """Normalize every ordinary malformed-input failure to one codec error."""

    try:
        return _decode_phase2b_scorer_bundle_v1_impl(bundle)
    except Phase2BScorerBundleErrorV1:
        raise
    except (
        AttributeError,
        KeyError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        UnicodeError,
        ValueError,
        struct.error,
    ) as exc:
        raise Phase2BScorerBundleErrorV1(
            f"scorer bundle decode rejected: {type(exc).__name__}"
        ) from exc


def _canonical_score_mechanics_report_v1(value: object) -> bytes:
    """Canonicalize one exact scoring result without JSON floats."""

    if type(value) is not Unsealed960PredictionScoringMechanicsV2:
        _fail("score report requires exact successful scoring mechanics")
    return _canonical_json_bytes_v1(
        {
            "actual_scoring_claimed": False,
            "formal_gate_evaluation_claimed": False,
            "report_kind": "PHASE2B_SCORE_MECHANICS_V1",
            "result": _report_primitive_v1(value),
        }
    ) + b"\n"


def _canonical_available_gate_mechanics_report_v1(value: object) -> bytes:
    """Canonicalize one exact available-gate result without JSON floats."""

    if type(value) is not Unsealed960AvailableGateMechanicsV2:
        _fail("available-gate report requires exact successful mechanics")
    return _canonical_json_bytes_v1(
        {
            "actual_scoring_claimed": False,
            "formal_gate_evaluation_claimed": False,
            "report_kind": "PHASE2B_AVAILABLE_GATE_MECHANICS_V1",
            "result": _report_primitive_v1(value),
        }
    ) + b"\n"


def evaluate_phase2b_scorer_bundle_v1(
    decoded: DecodedPhase2BScorerBundleV1,
) -> EvaluatedPhase2BScorerBundleV1:
    """Invoke scorer, replay validator, and gate evaluator exactly once each."""

    if type(decoded) is not DecodedPhase2BScorerBundleV1:
        _fail("bundle evaluator requires exact privately issued decoded DTO")
    try:
        supplied_bytes = object.__getattribute__(decoded, "bundle_bytes")
        supplied_id = object.__getattribute__(decoded, "bundle_id")
        supplied_sha = object.__getattribute__(decoded, "bundle_sha256")
    except AttributeError as exc:
        _fail(f"decoded bundle integrity surface is absent: {exc}")
    if (
        type(supplied_bytes) is not bytes
        or type(supplied_id) is not str
        or type(supplied_sha) is not str
    ):
        _fail("decoded bundle integrity surface exact types differ")
    expected_sha = hashlib.sha256(supplied_bytes).hexdigest()
    expected_id = _BUNDLE_ID_PREFIX_V1 + expected_sha
    if supplied_sha != expected_sha or supplied_id != expected_id:
        _fail("decoded bundle integrity identity differs from its bytes")
    fresh = decode_phase2b_scorer_bundle_v1(supplied_bytes)
    if fresh.bundle_id != supplied_id or fresh.bundle_sha256 != supplied_sha:
        _fail("decoded bundle integrity identity differs from fresh re-decode")
    decoded = fresh
    scoring_candidate = score_unsealed_960_prediction_scoring_mechanics_v2(
        prediction_archive=decoded.prediction_archive_bytes,
        structural_receipt=decoded.structural_receipt,
        structural_evaluation=decoded.structural_evaluation,
        partition_manifest=decoded.partition_manifest,
        answer_manifest=decoded.answer_manifest,
        revealed_answer_manifest_sha256=decoded.answer_manifest.answer_manifest_sha256,
        answer_commitment_salt=decoded.answer_commitment_salt,
        salted_answer_commitment_sha256=decoded.answer_commitment_sha256,
    )
    if type(scoring_candidate) is not Unsealed960PredictionScoringMechanicsV2:
        if type(scoring_candidate) is Unsealed960PredictionScoringRejectionV2:
            _fail(f"score mechanics rejected: {scoring_candidate.reason.value}")
        _fail("score mechanics returned a nonexact type")
    scoring = scoring_candidate
    replay_candidate = validate_actual_unsealed_960_replay_input_contract_v2(
        gate_input_manifest=decoded.gate_input_manifest,
        answer_manifest=decoded.answer_manifest,
        revealed_gate_input_manifest_sha256=(
            decoded.gate_input_manifest.gate_input_manifest_sha256
        ),
        gate_input_commitment_salt=decoded.gate_input_commitment_salt,
        salted_gate_input_commitment_sha256=decoded.gate_input_commitment_sha256,
    )
    if type(replay_candidate) is not ActualUnsealed960ReplayInputContractV2:
        if type(replay_candidate) is ActualUnsealed960ReplayInputContractRejectionV2:
            _fail(f"replay input contract rejected: {replay_candidate.reason.value}")
        _fail("replay validator returned a nonexact type")
    replay = replay_candidate
    gate_candidate = evaluate_unsealed_960_available_gate_mechanics_v2(
        scoring_mechanics=scoring,
        replay_input_contract=replay,
        gate_input_manifest=decoded.gate_input_manifest,
    )
    if type(gate_candidate) is not Unsealed960AvailableGateMechanicsV2:
        if type(gate_candidate) is Unsealed960AvailableGateMechanicsRejectionV2:
            _fail(f"available-gate mechanics rejected: {gate_candidate.reason.value}")
        _fail("available-gate evaluator returned a nonexact type")
    gate = gate_candidate
    score_report = _canonical_score_mechanics_report_v1(scoring)
    gate_report = _canonical_available_gate_mechanics_report_v1(gate)
    score_id = SCORE_REPORT_ID_PREFIX_V1 + hashlib.sha256(
        SCORE_REPORT_DOMAIN_V1 + score_report
    ).hexdigest()
    gate_id = GATE_REPORT_ID_PREFIX_V1 + hashlib.sha256(
        GATE_REPORT_DOMAIN_V1 + gate_report
    ).hexdigest()
    value = object.__new__(EvaluatedPhase2BScorerBundleV1)
    for name, item in (
        ("scoring_mechanics", scoring),
        ("replay_input_contract", replay),
        ("available_gate_mechanics", gate),
        ("score_report_bytes", score_report),
        ("score_report_id", score_id),
        ("available_gate_report_bytes", gate_report),
        ("available_gate_report_id", gate_id),
    ):
        object.__setattr__(value, name, item)
    return value


__all__ = (
    "PHASE2B_SCORER_BUNDLE_V1_FILENAME",
    "PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME",
    "PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME",
    "PHASE2B_SCORER_BUNDLE_V1_VERSION",
    "PHASE2B_SCORER_BUNDLE_V1_MAGIC",
    "PHASE2B_SCORING_SOURCE_SHA256_V1",
    "PHASE2B_SCORER_ACTOR_PROFILE_CONTENT_ID_V1",
    "PHASE2B_SCORER_ACTOR_PROFILE_CANONICAL_SHA256_V1",
    "PHASE2B_C1_OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V1",
    "SCORE_REPORT_ID_PREFIX_V1",
    "SCORE_REPORT_DOMAIN_V1",
    "GATE_REPORT_ID_PREFIX_V1",
    "GATE_REPORT_DOMAIN_V1",
    "MAXIMUM_PHASE2B_SCORER_BUNDLE_BYTES_V1",
    "MAXIMUM_PHASE2B_SCORER_METADATA_BYTES_V1",
    "SCORER_BUNDLE_FILENAME_V1",
    "SCORE_MECHANICS_REPORT_FILENAME_V1",
    "AVAILABLE_GATE_MECHANICS_REPORT_FILENAME_V1",
    "MAXIMUM_SCORER_BUNDLE_BYTES_V1",
    "Phase2BScorerBundleErrorV1",
    "DecodedPhase2BScorerBundleV1",
    "EvaluatedPhase2BScorerBundleV1",
    "Phase2BScorerBundleEvaluationV1",
    "encode_phase2b_scorer_bundle_v1",
    "decode_phase2b_scorer_bundle_v1",
    "evaluate_phase2b_scorer_bundle_v1",
)
