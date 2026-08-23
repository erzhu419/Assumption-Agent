"""Producer-free Package 0 raw mechanics for a future Phase-2B C1 run.

This module verifies one fully presented Package 0 byte graph against the
committed authoritative-evidence schema catalog.  The verification is local,
deterministic, and producer-free: it accepts only exact raw bytes and mandatory
out-of-band byte pins.  It does not generate keys, signatures, identifiers,
events, receipts, checkpoints, anchors, time, or operational evidence.

Successful verification establishes only the mechanics of the presented
Package 0 graph.  It does not establish R10, authority currentness, external
custody or durability, non-equivocation, execution, an actual-960 run, or C1.
The separately named real verifier remains exactly NOT_IMPLEMENTED.
"""

from __future__ import annotations

import builtins as _builtins
from copy import deepcopy as _deepcopy
from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from types import FunctionType, ModuleType
from typing import Final, NoReturn

from . import phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1 as _catalog
from . import rfc8785_v1 as _rfc8785

try:  # Absence of the backend is a fail-closed verification error.
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - exercised by an isolated import KAT.
    _InvalidSignature = None
    _Ed25519PublicKey = None


SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-durable-signed-custodian-"
    "ledger-and-signer-chain/1"
)
STATUS_V1: Final = "VERIFIED_PRESENTED_PACKAGE0_MECHANICS_NOT_R10"
CLAIM_LEVEL_V1: Final = (
    "PACKAGE0_RAW_GRAPH_AND_SIGNATURE_MECHANICS_ONLY_NOT_GATE_ADMISSION_"
    "NOT_R10_NOT_AUTHORITY_NOT_CURRENTNESS_NOT_DURABILITY_NOT_"
    "NON_EQUIVOCATION_NOT_EXECUTION_NOT_ACTUAL_NOT_C1"
)
CANONICALIZATION_V1: Final = "RFC8785_EXACT_BYTES_NO_TRAILING_LF"
REAL_VERIFICATION_STATUS_V1: Final = "NOT_IMPLEMENTED"

INSPECTED_CATALOG_SOURCE_SHA256_V1: Final = (
    "ead2e2e5a7b03a804148fbd329e67561c1367375b2c05bf4ab2cd691f20487c6"
)
INSPECTED_CATALOG_CONFIG_SHA256_V1: Final = (
    "62f46cfba8a86f7ad969a7ea584c1828f0b34bb1d95571b1cc64259ac01a588a"
)
INSPECTED_RFC8785_SOURCE_SHA256_V1: Final = (
    "16dcc6524a1c4a507edb9f85455c72b00fbc6c0c1b67471f03b67b2f25d32654"
)

SIGNATURE_ALGORITHM_V1: Final = "Ed25519"
DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-detached-signature/1"
)
DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-authoritative-evidence-"
    "detached-signature-preimage/1"
)
SIGNER_REGISTRY_STATUS_V1: Final = "PRESENTED_UNVERIFIED"
SIGNER_REGISTRY_CLAIM_LEVEL_V1: Final = _catalog.PACKAGE_OUTER_CLAIM_LEVEL_V1
SIGNER_REGISTRY_EPOCH_V1: Final = "1"
STAGE_EVENT_STATUS_V1: Final = "PRESENTED_UNVERIFIED"
STAGE_EVENT_CLAIM_LEVEL_V1: Final = _catalog.PACKAGE_OUTER_CLAIM_LEVEL_V1
STAGE_EVENT_OCCURRED_V1: Final = True
STAGE_EVENT_FIRST_FAILURE_CODE_V1: Final = "NONE"
PACKAGE0_SHALLOW_STAGE_SCOPE_V1: Final = (
    "PACKAGE0_SHALLOW_NONRECURSIVE_STAGE_ENVELOPE_BINDING_V1"
)

CAMPAIGN_ID_PREFIX_V1: Final = "phase2b_actual_unsealed_960_campaign_v1_"
ATTEMPT_ID_PREFIX_V1: Final = "phase2b_actual_unsealed_960_attempt_v1_"
CAMPAIGN_ID_FULLMATCH_ASCII_REGEX_V1: Final = (
    r"phase2b_actual_unsealed_960_campaign_v1_[0-9a-f]{64}"
)
ATTEMPT_ID_FULLMATCH_ASCII_REGEX_V1: Final = (
    r"phase2b_actual_unsealed_960_attempt_v1_[0-9a-f]{64}"
)
GENERIC_CONTENT_IDENTIFIER_FULLMATCH_ASCII_REGEX_V1: Final = (
    r"[a-z0-9_]+_[0-9a-f]{64}"
)
MAXIMUM_EXACT_BYTES_PARAMETER_OR_TUPLE_ITEM_BYTES_V1: Final = 16_777_216
MAXIMUM_JSON_SAFE_INTEGER_V1: Final = (1 << 53) - 1

LEDGER_EVENT_ENVELOPE_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-signed-stage-event/1"
)
LEDGER_EVENT_BODY_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-custodian-ledger-event/1"
)
LEDGER_EVENT_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-custodian-ledger-event-"
    "signature-preimage/1"
)
LEDGER_EVENT_ID_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/SIGNED_STAGE_EVENT/ID/V1\0"
)
LEDGER_EVENT_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_signed_stage_event_v1_"
)
LEDGER_EVENT_CHAIN_ROOT_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/"
    b"SIGNED_STAGE_EVENT_CHAIN/ROOT/V1\0"
)
LEDGER_EVENT_CHAIN_ROOT_PAIR_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "signed_stage_event_id",
    "signed_stage_event_payload_sha256",
)
LEDGER_EVENT_CHAIN_ROOT_REQUIRED_CARDINALITY_V1: Final = 10
LEDGER_EVENT_CHAIN_ROOT_PREIMAGE_PROFILE_V1: Final = (
    "NUL_TERMINATED_DOMAIN_BYTES_PLUS_RFC8785_EXACT_ARRAY_OF_10_"
    "SIGNED_STAGE_EVENT_ID_AND_RAW_PAYLOAD_SHA256_OBJECTS_V1"
)
LEDGER_EVENT_CHAIN_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_signed_stage_event_chain_v1_"
)
LEDGER_EVENT_CHAIN_ID_FORMULA_V1: Final = (
    "chain_id_prefix || chain_root_sha256_without_sha256_colon"
)

TERMINAL_STAGE_DISPOSITIONS_IN_EXACT_ORDER_V1: Final = (
    "REJECTED",
    "INVALIDATED",
    "COMPLETE_FAIL",
    "COMPLETE_PASS",
)
TERMINAL_SUBJECT_ARTIFACT_KINDS_IN_EXACT_ORDER_V1: Final = (
    "authoritative_evidence_package_15_v1",
    "authoritative_evidence_package_16_v1",
)
TERMINAL_SUBJECT_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1: Final = tuple(
    _catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[ordinal][
        "content_id_prefix"
    ]
    for ordinal in (15, 16)
)

VERIFIED_SIGNATURE_BREAKDOWN_IN_EXACT_ORDER_V1: Final = (
    ("package0_outer", 4),
    ("package0_ledger_events", 10),
    ("package0_referenced_artifacts", 6),
    ("signer_registry_oob_root", 2),
    ("stage_subject_events", 10),
    ("package15_shallow_outer", 1),
    ("package16_shallow_outer", 2),
)
PACKAGE0_OWNED_SIGNATURE_COUNT_V1: Final = 20
TOTAL_VERIFIED_SIGNATURE_COUNT_V1: Final = 35
STAGE_SUBJECT_FORBIDDEN_FORWARD_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1: Final = (
    _catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0]["content_id_prefix"],
    _catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[17]["content_id_prefix"],
    _catalog.BUNDLE_INDEX_SCHEMA_OBJECT_V1["content_id_prefix"],
    LEDGER_EVENT_ID_PREFIX_V1,
    LEDGER_EVENT_CHAIN_ID_PREFIX_V1,
    *tuple(
        descriptor["content_id_prefix"]
        for descriptor in _catalog.STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    ),
    *tuple(
        contract["record_content_id_prefix"]
        for contract in _catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0][
            "nested_record_binding_contracts_in_exact_order"
        ][1:6]
    ),
)
STAGE_SUBJECT_FORBIDDEN_FORWARD_ARTIFACT_KINDS_IN_EXACT_ORDER_V1: Final = (
    _catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0][
        "signature_preimage_artifact_kind"
    ],
    _catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[17][
        "signature_preimage_artifact_kind"
    ],
    _catalog.BUNDLE_INDEX_SCHEMA_OBJECT_V1["signature_preimage_artifact_kind"],
    "signed_stage_event_chain_v1",
    *tuple(
        descriptor["signature_preimage_artifact_kind"]
        for descriptor in _catalog.STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    ),
    *tuple(f"signed_custodian_stage_event_{index}_v1" for index in range(10)),
    *tuple(
        contract["signature_preimage_artifact_kind"]
        for contract in _catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0][
            "nested_record_binding_contracts_in_exact_order"
        ][1:6]
    ),
)

POSITIONAL_PARAMETERS_IN_EXACT_ORDER_V1: Final = (
    ("catalog_payload", "EXACT_BUILTIN_BYTES"),
    ("raw_signer_registry_payload", "EXACT_BUILTIN_BYTES"),
    ("raw_package0_envelope_payload", "EXACT_BUILTIN_BYTES"),
    (
        "raw_package0_ledger_event_payloads_in_exact_order",
        "EXACT_BUILTIN_TUPLE_OF_10_EXACT_BUILTIN_BYTES",
    ),
    (
        "raw_package0_stage_subject_event_payloads_in_exact_order",
        "EXACT_BUILTIN_TUPLE_OF_10_EXACT_BUILTIN_BYTES",
    ),
    ("raw_store_if_absent_receipt_payload", "EXACT_BUILTIN_BYTES"),
    ("raw_logical_time_token_payload", "EXACT_BUILTIN_BYTES"),
    ("raw_governance_checkpoint_payload", "EXACT_BUILTIN_BYTES"),
    ("raw_custodian_checkpoint_payload", "EXACT_BUILTIN_BYTES"),
    ("raw_external_anchor_payload", "EXACT_BUILTIN_BYTES"),
    ("raw_package15_envelope_payload", "EXACT_BUILTIN_BYTES"),
    ("raw_package16_envelope_payload", "EXACT_BUILTIN_BYTES"),
)
KEYWORD_ONLY_PARAMETERS_IN_EXACT_ORDER_V1: Final = (
    "out_of_band_expected_catalog_id_bytes",
    "out_of_band_expected_catalog_payload_sha256_digest",
    "out_of_band_expected_signer_registry_id_bytes",
    "out_of_band_expected_signer_registry_payload_sha256_digest",
    "out_of_band_expected_signer_registry_entries_root_sha256_digest",
    "out_of_band_expected_governance_root_key_id_bytes",
    "out_of_band_expected_governance_root_public_key_bytes",
    "out_of_band_expected_custodian_root_key_id_bytes",
    "out_of_band_expected_custodian_root_public_key_bytes",
    "out_of_band_expected_readiness_dag_id_bytes",
    "out_of_band_expected_readiness_dag_payload_sha256_digest",
    "out_of_band_expected_progression_plan_payload_sha256_digest",
    "out_of_band_expected_gap_inventory_source_sha256_digest",
    "out_of_band_expected_statistical_validation_source_sha256_digest",
    "expected_campaign_id_bytes",
    "expected_attempt_id_bytes",
)

MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "exact_raw_bytes_verified",
    "rfc8785_canonical_bytes_verified",
    "content_ids_verified",
    "raw_payload_sha256s_verified",
    "signature_contracts_verified",
    "ledger_chain_verified",
    "stage_subject_binding_verified",
    "referenced_artifact_graph_verified",
    "package0_lineage_verified",
    "terminal_p15_p16_binding_verified",
    "predecessor_absence_verified",
    "private_lineage_verified",
)
NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "package15_verified",
    "package16_verified",
    "gate_admission_verified",
    "r10_satisfied",
    "authority_currentness_verified",
    "owner_non_revocation_verified",
    "external_key_custody_verified",
    "external_durability_verified",
    "external_non_equivocation_verified",
    "global_anti_fork_verified",
    "execution_authorized",
    "actual_960_verified",
    "c1_satisfied",
)


class Package0MechanicsDispositionV1(str, Enum):
    VERIFIED_PRESENTED_PACKAGE0_MECHANICS_NOT_R10 = (
        "VERIFIED_PRESENTED_PACKAGE0_MECHANICS_NOT_R10"
    )


class Package0MechanicsReasonV1(str, Enum):
    EXACT_PACKAGE0_RAW_GRAPH_VERIFIED = "EXACT_PACKAGE0_RAW_GRAPH_VERIFIED"


class Phase2BActualUnsealed960Package0MechanicsErrorV1(ValueError):
    """Raised when the exact presented Package 0 graph fails closed."""


@dataclass(frozen=True, slots=True, init=False)
class VerifiedPhase2BActualUnsealed960Package0MechanicsV1:
    schema_version: str
    status: str
    claim_level: str
    disposition: Package0MechanicsDispositionV1
    reason: Package0MechanicsReasonV1
    catalog_id: str
    catalog_payload_sha256: str
    package_schema_id: str
    package_id: str
    package_payload_sha256: str
    signer_registry_id: str
    signer_registry_payload_sha256: str
    signer_registry_entries_root_sha256: str
    campaign_id: str
    attempt_id: str
    chain_root_sha256: str
    terminal_signed_stage_event_id: str
    terminal_signed_stage_event_payload_sha256: str
    ledger_event_count: int
    stage_subject_event_count: int
    referenced_artifact_count: int
    total_signature_count: int
    exact_raw_bytes_verified: bool
    rfc8785_canonical_bytes_verified: bool
    content_ids_verified: bool
    raw_payload_sha256s_verified: bool
    signature_contracts_verified: bool
    ledger_chain_verified: bool
    stage_subject_binding_verified: bool
    referenced_artifact_graph_verified: bool
    package0_lineage_verified: bool
    terminal_p15_p16_binding_verified: bool
    predecessor_absence_verified: bool
    private_lineage_verified: bool
    package15_verified: bool
    package16_verified: bool
    gate_admission_verified: bool
    r10_satisfied: bool
    authority_currentness_verified: bool
    owner_non_revocation_verified: bool
    external_key_custody_verified: bool
    external_durability_verified: bool
    external_non_equivocation_verified: bool
    global_anti_fork_verified: bool
    execution_authorized: bool
    actual_960_verified: bool
    c1_satisfied: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("verified Package 0 mechanics results are privately issued")

    def __copy__(self, _type_error: type[TypeError] = TypeError) -> NoReturn:
        raise _type_error("verified Package 0 mechanics results cannot be copied")

    def __deepcopy__(
        self,
        memo: object,
        _type_error: type[TypeError] = TypeError,
    ) -> NoReturn:
        del memo
        raise _type_error(
            "verified Package 0 mechanics results cannot be deep-copied"
        )

    def __reduce_ex__(
        self,
        protocol: object,
        _type_error: type[TypeError] = TypeError,
    ) -> NoReturn:
        del protocol
        raise _type_error("verified Package 0 mechanics results cannot be serialized")


RESULT_FIELDS_IN_EXACT_ORDER_V1: Final = tuple(
    VerifiedPhase2BActualUnsealed960Package0MechanicsV1.__dataclass_fields__
)
RESULT_FIELD_TYPES_IN_EXACT_ORDER_V1: Final = tuple(
    (
        field_name,
        {
            "Package0DispositionV1": "Package0MechanicsDispositionV1",
            "Package0ReasonV1": "Package0MechanicsReasonV1",
        }.get(field_type, field_type),
    )
    for field_name, field_type in _catalog.PACKAGE0_VERIFIED_RESULT_FIELDS_IN_EXACT_ORDER_V1
)


def _capture_dependency_function_graph_v1(
    module: ModuleType,
) -> dict[str, FunctionType]:
    """Clone a dependency's local function graph into private globals."""

    namespace = dict(vars(module))
    originals = {
        name: value
        for name, value in namespace.items()
        if type(value) is FunctionType and value.__module__ == module.__name__
    }
    private_namespace = dict(namespace)
    private_namespace["__builtins__"] = dict(vars(_builtins))
    provisional: dict[str, FunctionType] = {}
    provisional_by_id: dict[int, FunctionType] = {}
    for name, function in originals.items():
        clone = FunctionType(
            function.__code__,
            private_namespace,
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )
        clone.__kwdefaults__ = function.__kwdefaults__
        provisional[name] = clone
        provisional_by_id[id(function)] = clone
    private_namespace.update(provisional)

    def captured_default(value: object) -> object:
        return provisional_by_id.get(id(value), value)

    captured: dict[str, FunctionType] = {}
    for name, function in originals.items():
        defaults = function.__defaults__
        clone = FunctionType(
            function.__code__,
            private_namespace,
            function.__name__,
            None
            if defaults is None
            else tuple(captured_default(item) for item in defaults),
            function.__closure__,
        )
        clone.__kwdefaults__ = (
            None
            if function.__kwdefaults__ is None
            else {
                key: captured_default(value)
                for key, value in function.__kwdefaults__.items()
            }
        )
        captured[name] = clone
    private_namespace.update(captured)
    return captured


def _load_exact_rfc8785_function_graph_v1(
    imported_module: ModuleType,
) -> dict[str, FunctionType]:
    """Load the frozen local dependency bytes into a private module graph."""

    source_path = getattr(imported_module, "__file__", None)
    if type(source_path) is not str or not source_path:
        raise RuntimeError("Package0 RFC8785 source path is unavailable")
    try:
        with _builtins.open(source_path, "rb") as handle:
            source_bytes = handle.read()
    except OSError as exc:
        raise RuntimeError("Package0 RFC8785 source bytes are unavailable") from exc
    if hashlib.sha256(source_bytes).hexdigest() != INSPECTED_RFC8785_SOURCE_SHA256_V1:
        raise RuntimeError("Package0 RFC8785 source bytes differ")

    private_module = ModuleType(
        "hegel_machine._phase2b_actual_unsealed_960_package0_frozen_rfc8785_v1"
    )
    private_namespace = vars(private_module)
    private_namespace["__file__"] = source_path
    private_namespace["__package__"] = "hegel_machine"
    private_namespace["__builtins__"] = dict(vars(_builtins))
    code = _builtins.compile(source_bytes, source_path, "exec", dont_inherit=True)
    _builtins.exec(code, private_namespace)
    return _capture_dependency_function_graph_v1(private_module)


_RFC8785_FUNCTIONS_V1 = _load_exact_rfc8785_function_graph_v1(_rfc8785)
_CAPTURED_CANONICALIZE_V1 = _RFC8785_FUNCTIONS_V1[
    "canonicalize_rfc8785_v1"
]
_CAPTURED_DECODE_V1 = _RFC8785_FUNCTIONS_V1[
    "decode_and_require_canonical_rfc8785_v1"
]
del _RFC8785_FUNCTIONS_V1
del _load_exact_rfc8785_function_graph_v1
del _capture_dependency_function_graph_v1


def _assert_catalog_public_surface_matches_exact_payload_v1() -> None:
    """Reject pre-import rebinding or mutation of the inspected catalog."""

    raw = _catalog.CATALOG_PAYLOAD_V1
    if (
        type(raw) is not bytes
        or hashlib.sha256(raw).hexdigest()
        != INSPECTED_CATALOG_CONFIG_SHA256_V1
        or _catalog.CATALOG_PAYLOAD_SHA256_V1
        != INSPECTED_CATALOG_CONFIG_SHA256_V1
    ):
        raise RuntimeError("Package0 inspected catalog payload pin differs")
    try:
        decoded = _CAPTURED_DECODE_V1(raw)
    except Exception as exc:
        raise RuntimeError("Package0 inspected catalog payload is not canonical") from exc
    if (
        type(decoded) is not dict
        or frozenset(decoded) != frozenset(("schema_version", "body", "catalog_id"))
        or type(decoded["body"]) is not dict
        or _CAPTURED_CANONICALIZE_V1(decoded) != raw
        or _catalog.CATALOG_ID_V1 != decoded["catalog_id"]
    ):
        raise RuntimeError("Package0 inspected catalog envelope differs")
    body = decoded["body"]

    def json_projection(value: object) -> object:
        if type(value) is tuple:
            return [json_projection(item) for item in value]
        if type(value) is list:
            return [json_projection(item) for item in value]
        if type(value) is dict:
            return {key: json_projection(item) for key, item in value.items()}
        return value

    surface_pairs = (
        (
            _catalog.SIGNER_REGISTRY_SCHEMA_OBJECT_V1,
            body["signer_registry_schema_object"],
        ),
        (
            _catalog.GATE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
            body["gate_schema_objects_in_exact_order"],
        ),
        (
            _catalog.STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
            body["stage_schema_objects_in_exact_order"],
        ),
        (
            _catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
            body["package_schema_objects_in_exact_order"],
        ),
        (
            _catalog.BUNDLE_INDEX_SCHEMA_OBJECT_V1,
            body["bundle_index_schema_object"],
        ),
        (
            _catalog.FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1,
            body["future_real_package0_verifier_raw_abi"],
        ),
    )
    for public_surface, payload_surface in surface_pairs:
        if _CAPTURED_CANONICALIZE_V1(
            json_projection(public_surface)
        ) != _CAPTURED_CANONICALIZE_V1(payload_surface):
            raise RuntimeError("Package0 catalog public schema surface differs")

    result_rows = tuple(
        tuple(row)
        for row in body["future_real_package0_verified_result_contract"][
            "fields_in_exact_order"
        ]
    )
    stage_names = tuple(
        item["stage"] for item in body["stage_schema_objects_in_exact_order"]
    )
    dependencies = {
        item["dependency_name"]: item
        for item in body["inspected_dependencies_in_exact_order"]
    }
    if (
        tuple(_catalog.PACKAGE0_VERIFIED_RESULT_FIELDS_IN_EXACT_ORDER_V1)
        != result_rows
        or tuple(_catalog.STAGE_NAMES_IN_EXACT_ORDER_V1) != stage_names
        or _catalog.DAG_ROOT_V1 != body["interleaved_dag_root"]
        or _catalog.PACKAGE_OUTER_CLAIM_LEVEL_V1
        != body["package_schema_objects_in_exact_order"][0][
            "body_required_literals"
        ]["claim_level"]
        or _catalog.INSPECTED_READINESS_DAG_ID_V1
        != dependencies["readiness_dag_config"]["content_id_or_none"]
        or _catalog.INSPECTED_READINESS_DAG_CONFIG_SHA256_V1
        != dependencies["readiness_dag_config"]["sha256"]
        or _catalog.INSPECTED_PROGRESSION_PLAN_SHA256_V1
        != dependencies["c1_c4_progression_plan"]["sha256"]
        or _catalog.INSPECTED_GAP_INVENTORY_SOURCE_SHA256_V1
        != dependencies["legacy_gap_inventory_source"]["sha256"]
        or _catalog.INSPECTED_STATISTICAL_VALIDATION_SOURCE_SHA256_V1
        != dependencies["statistical_validation_source"]["sha256"]
    ):
        raise RuntimeError("Package0 catalog public constant surface differs")

    package_schemas = _catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    nested_contracts = package_schemas[0][
        "nested_record_binding_contracts_in_exact_order"
    ]
    actual_signature_breakdown = (
        ("package0_outer", package_schemas[0]["signature_cardinality"]),
        (
            "package0_ledger_events",
            nested_contracts[0]["nested_signature_cardinality"],
        ),
        (
            "package0_referenced_artifacts",
            sum(
                item["signature_cardinality"]
                for item in nested_contracts[1:6]
            ),
        ),
        (
            "signer_registry_oob_root",
            _catalog.SIGNER_REGISTRY_SCHEMA_OBJECT_V1["signature_cardinality"],
        ),
        (
            "stage_subject_events",
            sum(
                item["signature_cardinality"]
                for item in _catalog.STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
            ),
        ),
        ("package15_shallow_outer", package_schemas[15]["signature_cardinality"]),
        ("package16_shallow_outer", package_schemas[16]["signature_cardinality"]),
    )
    if (
        actual_signature_breakdown
        != VERIFIED_SIGNATURE_BREAKDOWN_IN_EXACT_ORDER_V1
        or sum(count for _, count in actual_signature_breakdown)
        != TOTAL_VERIFIED_SIGNATURE_COUNT_V1
        or sum(
            count
            for name, count in actual_signature_breakdown
            if name.startswith("package0_")
        )
        != PACKAGE0_OWNED_SIGNATURE_COUNT_V1
    ):
        raise RuntimeError("Package0 catalog signature cardinality differs")


_assert_catalog_public_surface_matches_exact_payload_v1()
del _assert_catalog_public_surface_matches_exact_payload_v1


def canonical_phase2b_actual_unsealed_960_package0_mechanics_bytes_v1(
    value: object,
    _canonicalize: FunctionType = _CAPTURED_CANONICALIZE_V1,
) -> bytes:
    """Return exact RFC8785 bytes using the definition-time function graph."""

    return _canonicalize(value)


CONTRACT_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-durable-signed-custodian-"
    "ledger-and-signer-chain-contract/1"
)
CONTRACT_ID_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/PACKAGE0_MECHANICS/CONTRACT/ID/V1\0"
)
CONTRACT_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_package0_mechanics_contract_v1_"
)
CONTRACT_BODY_V1: Final = {
    "schema_version": CONTRACT_SCHEMA_VERSION_V1,
    "status": "LOCAL_PRODUCER_FREE_MECHANICS_IMPLEMENTED_NOT_R10",
    "claim_level": CLAIM_LEVEL_V1,
    "canonicalization": CANONICALIZATION_V1,
    "inspected_catalog_id": _catalog.CATALOG_ID_V1,
    "inspected_catalog_payload_sha256": (
        "sha256:" + _catalog.CATALOG_PAYLOAD_SHA256_V1
    ),
    "inspected_catalog_source_sha256": INSPECTED_CATALOG_SOURCE_SHA256_V1,
    "inspected_catalog_config_sha256": INSPECTED_CATALOG_CONFIG_SHA256_V1,
    "inspected_rfc8785_source_sha256": INSPECTED_RFC8785_SOURCE_SHA256_V1,
    "public_presented_verifier": (
        "verify_phase2b_actual_unsealed_960_durable_signed_custodian_"
        "ledger_and_signer_chain_mechanics_v1"
    ),
    "public_real_verifier": (
        "verify_real_phase2b_actual_unsealed_960_durable_signed_custodian_"
        "ledger_and_signer_chain_v1"
    ),
    "real_verification_status": REAL_VERIFICATION_STATUS_V1,
    "positional_parameters_in_exact_order": [
        [name, wire_type]
        for name, wire_type in POSITIONAL_PARAMETERS_IN_EXACT_ORDER_V1
    ],
    "mandatory_keyword_only_parameters_in_exact_order": list(
        KEYWORD_ONLY_PARAMETERS_IN_EXACT_ORDER_V1
    ),
    "catalog_frozen_package0_raw_abi_exact_match": True,
    "catalog_frozen_package0_result_field_names_exact_match": True,
    "catalog_public_surface_cross_bound_to_exact_payload": True,
    "catalog_signature_cardinality_breakdown_exact": True,
    "rfc8785_dependency_capture_profile": (
        "DEFINITION_TIME_READ_ONLY_EXACT_LOCAL_SOURCE_BYTES_SHA256_"
        "THEN_PRIVATE_REEXECUTION_AND_FUNCTION_GRAPH_CAPTURE"
    ),
    "rfc8785_dependency_source_path_is_caller_input": False,
    "rfc8785_dependency_source_check_claims_currentness_or_authority": False,
    "signer_registry_required_literals": {
        "schema_version": (
            "hegel-machine-phase2b-actual-unsealed-960-signer-registry/1"
        ),
        "status": SIGNER_REGISTRY_STATUS_V1,
        "claim_level": SIGNER_REGISTRY_CLAIM_LEVEL_V1,
        "registry_epoch": SIGNER_REGISTRY_EPOCH_V1,
    },
    "signer_registry_entry_ordinal_profile": (
        "EXACT_ASCII_DECIMAL_ENUM_MEMBER_0_THROUGH_5_MATCHING_THE_"
        "FROZEN_CATALOG_WIRE_TYPE"
    ),
    "signer_registry_control_domain_profile": (
        "EXACT_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
    ),
    "signer_registry_key_separation": (
        "SIX_REGISTRY_KEYS_AND_TWO_OOB_ROOT_KEYS_PAIRWISE_DISTINCT_"
        "BY_KEY_ID_AND_PUBLIC_KEY_WITH_REGISTRY_CONTROL_DOMAINS_DISTINCT"
    ),
    "stage_event_required_literal_profile": {
        "status": STAGE_EVENT_STATUS_V1,
        "claim_level": STAGE_EVENT_CLAIM_LEVEL_V1,
        "event_sequence": "EXACT_STAGE_ORDINAL_0_THROUGH_9",
        "stage_occurred": STAGE_EVENT_OCCURRED_V1,
        "first_failure_code": STAGE_EVENT_FIRST_FAILURE_CODE_V1,
        "nonterminal_disposition": None,
        "terminal_dispositions_in_exact_order": list(
            TERMINAL_STAGE_DISPOSITIONS_IN_EXACT_ORDER_V1
        ),
    },
    "stage_subject_verification_scope": PACKAGE0_SHALLOW_STAGE_SCOPE_V1,
    "raw_stage_predecessor_gate_evidence_verified": False,
    "raw_nonterminal_subject_artifacts_verified": False,
    "package15_full_verification_performed": False,
    "package16_full_verification_performed": False,
    "package15_package16_shallow_validation_profile": (
        "EXACT_ENVELOPE_AND_BODY_KEYSETS_REQUIRED_LITERALS_WIRE_TYPES_"
        "LINEAGE_PREDECESSOR_PREFIXES_CONTENT_ID_AND_SIGNATURES_WITHOUT_"
        "NESTED_RECORD_OR_GATE_DECISION_VERIFICATION"
    ),
    "terminal_subject_artifact_kinds_in_exact_order": list(
        TERMINAL_SUBJECT_ARTIFACT_KINDS_IN_EXACT_ORDER_V1
    ),
    "terminal_subject_content_id_prefixes_in_exact_order": list(
        TERMINAL_SUBJECT_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1
    ),
    "stage_subject_forbidden_forward_content_id_prefixes_in_exact_order": list(
        STAGE_SUBJECT_FORBIDDEN_FORWARD_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1
    ),
    "stage_subject_forbidden_forward_artifact_kinds_in_exact_order": list(
        STAGE_SUBJECT_FORBIDDEN_FORWARD_ARTIFACT_KINDS_IN_EXACT_ORDER_V1
    ),
    "local_stricter_input_constraints": {
        "scope": "LOCAL_STRICTER_MECHANICS_CONTRACT_NOT_FROZEN_CATALOG_REISSUE",
        "maximum_exact_bytes_parameter_or_tuple_item_bytes": (
            MAXIMUM_EXACT_BYTES_PARAMETER_OR_TUPLE_ITEM_BYTES_V1
        ),
        "maximum_json_safe_integer": MAXIMUM_JSON_SAFE_INTEGER_V1,
        "generic_content_identifier_fullmatch_ascii_regex": (
            GENERIC_CONTENT_IDENTIFIER_FULLMATCH_ASCII_REGEX_V1
        ),
        "campaign_id_fullmatch_ascii_regex": (
            CAMPAIGN_ID_FULLMATCH_ASCII_REGEX_V1
        ),
        "attempt_id_fullmatch_ascii_regex": ATTEMPT_ID_FULLMATCH_ASCII_REGEX_V1,
    },
    "ledger_event_signature_artifact_schema_version": (
        LEDGER_EVENT_ENVELOPE_SCHEMA_VERSION_V1
    ),
    "chain_root_domain_ascii": LEDGER_EVENT_CHAIN_ROOT_DOMAIN_V1.decode(
        "ascii"
    ),
    "chain_root_preimage_profile": (
        LEDGER_EVENT_CHAIN_ROOT_PREIMAGE_PROFILE_V1
    ),
    "chain_root_pair_fields_in_exact_order": list(
        LEDGER_EVENT_CHAIN_ROOT_PAIR_FIELDS_IN_EXACT_ORDER_V1
    ),
    "chain_root_required_cardinality": (
        LEDGER_EVENT_CHAIN_ROOT_REQUIRED_CARDINALITY_V1
    ),
    "chain_id_prefix": LEDGER_EVENT_CHAIN_ID_PREFIX_V1,
    "chain_id_formula": LEDGER_EVENT_CHAIN_ID_FORMULA_V1,
    "chain_id_formula_scope": (
        "LOCAL_STRICTER_MECHANICS_CONTRACT_NOT_FROZEN_CATALOG_REISSUE"
    ),
    "verified_signature_breakdown_in_exact_order": [
        [name, count]
        for name, count in VERIFIED_SIGNATURE_BREAKDOWN_IN_EXACT_ORDER_V1
    ],
    "package0_owned_signature_count": PACKAGE0_OWNED_SIGNATURE_COUNT_V1,
    "total_verified_signature_count": TOTAL_VERIFIED_SIGNATURE_COUNT_V1,
    "digest_representation": {
        "json_digest": "sha256:<64_lowerhex>",
        "content_id": "descriptor_prefix_plus_64_lowerhex",
        "signature_hex": "128_lowerhex",
        "public_key_hex": "64_lowerhex",
        "abi_digest_pin": "exact_32_raw_bytes",
        "abi_identifier_pin": "strict_ascii_bytes",
    },
    "mechanics_boolean_fields_in_exact_order": list(
        MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1
    ),
    "nonclaim_boolean_fields_required_false_in_exact_order": list(
        NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1
    ),
    "private_result_fields_in_exact_order": list(RESULT_FIELDS_IN_EXACT_ORDER_V1),
    "private_result_field_types_in_exact_order": [
        [name, field_type]
        for name, field_type in RESULT_FIELD_TYPES_IN_EXACT_ORDER_V1
    ],
    "private_result_type_name": (
        "VerifiedPhase2BActualUnsealed960Package0MechanicsV1"
    ),
    "private_result_require_callable_name": (
        "require_verified_phase2b_actual_unsealed_960_durable_signed_"
        "custodian_ledger_and_signer_chain_mechanics_v1"
    ),
    "private_result_status": STATUS_V1,
    "private_result_claim_level": CLAIM_LEVEL_V1,
    "private_result_success_disposition": (
        "VERIFIED_PRESENTED_PACKAGE0_MECHANICS_NOT_R10"
    ),
    "private_result_success_reason": "EXACT_PACKAGE0_RAW_GRAPH_VERIFIED",
    "private_result_issuer_only": True,
    "private_result_copy_deepcopy_pickle_and_mutation_allowed": False,
    "ed25519_verification_dependency": "cryptography>=41",
}
CONTRACT_ID_V1: Final = CONTRACT_ID_PREFIX_V1 + hashlib.sha256(
    CONTRACT_ID_DOMAIN_V1
    + canonical_phase2b_actual_unsealed_960_package0_mechanics_bytes_v1(
        CONTRACT_BODY_V1
    )
).hexdigest()
CONTRACT_PAYLOAD_V1: Final = (
    canonical_phase2b_actual_unsealed_960_package0_mechanics_bytes_v1(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION_V1,
            "body": CONTRACT_BODY_V1,
            "contract_id": CONTRACT_ID_V1,
        }
    )
)
CONTRACT_PAYLOAD_SHA256_V1: Final = hashlib.sha256(CONTRACT_PAYLOAD_V1).hexdigest()


def _build_private_boundary_v1():
    """Build a definition-time closed verifier and private result lineage."""

    result_type = VerifiedPhase2BActualUnsealed960Package0MechanicsV1
    error_type = Phase2BActualUnsealed960Package0MechanicsErrorV1
    disposition_type = Package0MechanicsDispositionV1
    reason_type = Package0MechanicsReasonV1
    expected_result_fields = tuple(RESULT_FIELDS_IN_EXACT_ORDER_V1)
    expected_catalog_payload = bytes(_catalog.CATALOG_PAYLOAD_V1)
    expected_catalog_id = str(_catalog.CATALOG_ID_V1)
    expected_catalog_sha = hashlib.sha256(expected_catalog_payload).digest()
    expected_catalog_sha_text = hashlib.sha256(expected_catalog_payload).hexdigest()
    expected_readiness_id = _catalog.INSPECTED_READINESS_DAG_ID_V1.encode(
        "ascii"
    )
    expected_readiness_sha = bytes.fromhex(
        _catalog.INSPECTED_READINESS_DAG_CONFIG_SHA256_V1
    )
    expected_plan_sha = bytes.fromhex(
        _catalog.INSPECTED_PROGRESSION_PLAN_SHA256_V1
    )
    expected_gap_sha = bytes.fromhex(
        _catalog.INSPECTED_GAP_INVENTORY_SOURCE_SHA256_V1
    )
    expected_statistical_sha = bytes.fromhex(
        _catalog.INSPECTED_STATISTICAL_VALIDATION_SOURCE_SHA256_V1
    )
    signer_schema = _deepcopy(_catalog.SIGNER_REGISTRY_SCHEMA_OBJECT_V1)
    gate_schemas = tuple(
        _deepcopy(item)
        for item in _catalog.GATE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    )
    stage_schemas = tuple(
        _deepcopy(item)
        for item in _catalog.STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    )
    package_schemas = tuple(
        _deepcopy(item)
        for item in _catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    )
    package0_schema = package_schemas[0]
    package15_schema = package_schemas[15]
    package16_schema = package_schemas[16]
    package0_nested_contracts = tuple(
        _deepcopy(item)
        for item in package0_schema[
            "nested_record_binding_contracts_in_exact_order"
        ]
    )
    ledger_contract = package0_nested_contracts[0]
    referenced_contracts = package0_nested_contracts[1:6]
    expected_catalog_dag_root = str(_catalog.DAG_ROOT_V1)
    expected_registry_roles = tuple(signer_schema["required_roles_in_exact_order"])
    expected_authorization_rows = tuple(
        tuple(row[field] for field in (
            "artifact_kind",
            "signature_ordinal",
            "role",
            "purpose",
            "key_epoch",
            "signature_domain_ascii",
        ))
        for row in signer_schema["required_authorization_rows_in_exact_order"]
    )
    stage_names = tuple(_catalog.STAGE_NAMES_IN_EXACT_ORDER_V1)
    terminal_dispositions = frozenset(
        TERMINAL_STAGE_DISPOSITIONS_IN_EXACT_ORDER_V1
    )
    terminal_subject_kinds = tuple(
        TERMINAL_SUBJECT_ARTIFACT_KINDS_IN_EXACT_ORDER_V1
    )
    terminal_subject_prefixes = tuple(
        TERMINAL_SUBJECT_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1
    )
    mechanics_fields = tuple(MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1)
    nonclaim_fields = tuple(NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1)
    canonicalize = _CAPTURED_CANONICALIZE_V1
    decode_canonical = _CAPTURED_DECODE_V1
    sha256 = hashlib.sha256
    public_key_type = _Ed25519PublicKey
    invalid_signature_type = _InvalidSignature
    type_builtin = type
    object_type = object
    bytes_type = bytes
    tuple_type = tuple
    list_type = list
    dict_type = dict
    str_type = str
    int_type = int
    bool_type = bool
    none_type = type(None)
    len_builtin = len
    tuple_builtin = tuple
    zip_builtin = zip
    all_builtin = all
    any_builtin = any
    frozenset_builtin = frozenset
    id_builtin = id
    getattr_builtin = getattr
    enumerate_builtin = enumerate
    set_builtin = set
    str_builtin = str
    abs_builtin = abs
    float_type = float
    exception_type = Exception
    unicode_decode_error_type = UnicodeDecodeError
    unicode_encode_error_type = UnicodeEncodeError
    type_error_type = TypeError
    value_error_type = ValueError
    expected_signature_algorithm = SIGNATURE_ALGORITHM_V1
    expected_detached_signature_schema = (
        DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
    )
    expected_registry_status = SIGNER_REGISTRY_STATUS_V1
    expected_registry_claim_level = SIGNER_REGISTRY_CLAIM_LEVEL_V1
    expected_registry_epoch = SIGNER_REGISTRY_EPOCH_V1
    expected_stage_status = STAGE_EVENT_STATUS_V1
    expected_stage_claim_level = STAGE_EVENT_CLAIM_LEVEL_V1
    expected_stage_occurred = STAGE_EVENT_OCCURRED_V1
    expected_stage_first_failure = STAGE_EVENT_FIRST_FAILURE_CODE_V1
    ledger_envelope_schema = LEDGER_EVENT_ENVELOPE_SCHEMA_VERSION_V1
    ledger_body_schema = LEDGER_EVENT_BODY_SCHEMA_VERSION_V1
    ledger_signature_preimage_schema = (
        LEDGER_EVENT_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
    )
    ledger_id_domain = bytes(LEDGER_EVENT_ID_DOMAIN_V1)
    ledger_id_prefix = str(LEDGER_EVENT_ID_PREFIX_V1)
    ledger_chain_root_domain = bytes(LEDGER_EVENT_CHAIN_ROOT_DOMAIN_V1)
    ledger_chain_id_prefix = str(LEDGER_EVENT_CHAIN_ID_PREFIX_V1)
    expected_status = str(STATUS_V1)
    expected_claim_level = str(CLAIM_LEVEL_V1)
    expected_schema_version = str(SCHEMA_VERSION_V1)
    expected_total_signature_count = TOTAL_VERIFIED_SIGNATURE_COUNT_V1
    forbidden_stage_subject_prefixes = tuple(
        STAGE_SUBJECT_FORBIDDEN_FORWARD_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1
    )
    forbidden_stage_subject_kinds = tuple(
        STAGE_SUBJECT_FORBIDDEN_FORWARD_ARTIFACT_KINDS_IN_EXACT_ORDER_V1
    )
    max_payload_bytes = MAXIMUM_EXACT_BYTES_PARAMETER_OR_TUPLE_ITEM_BYTES_V1
    safe_integer = MAXIMUM_JSON_SAFE_INTEGER_V1
    hex64_re = re.compile(r"[0-9a-f]{64}")
    hex128_re = re.compile(r"[0-9a-f]{128}")
    digest_re = re.compile(r"sha256:[0-9a-f]{64}")
    identifier_re = re.compile(GENERIC_CONTENT_IDENTIFIER_FULLMATCH_ASCII_REGEX_V1)
    campaign_re = re.compile(CAMPAIGN_ID_FULLMATCH_ASCII_REGEX_V1)
    attempt_re = re.compile(ATTEMPT_ID_FULLMATCH_ASCII_REGEX_V1)
    detached_signature_fields = frozenset(
        (
            "schema_version",
            "algorithm",
            "role",
            "purpose",
            "key_epoch",
            "signer_key_id",
            "signature_hex",
        )
    )
    ledger_envelope_fields = frozenset(
        ("schema_version", "body", "signed_stage_event_id", "signatures_in_exact_order")
    )
    ledger_body_fields = frozenset(
        (
            "schema_version",
            "catalog_id",
            "campaign_id",
            "attempt_id",
            "event_index",
            "stage",
            "predecessor_event_id_or_none",
            "payload_id",
            "payload_sha256",
            "signer_registry_id",
            "signer_key_id",
        )
    )
    registry: dict[int, tuple[object, tuple[object, ...]]] = {}

    def fail(code: str, detail: str) -> NoReturn:
        raise error_type(f"{code}: {detail}")

    def exact_bytes(
        value: object,
        label: str,
        *,
        length: int | None = None,
        nonempty: bool = True,
    ) -> bytes:
        if type_builtin(value) is not bytes_type:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_TYPE", f"{label} must be exact bytes")
        raw = value
        if nonempty and len_builtin(raw) == 0:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_TYPE", f"{label} must be nonempty")
        if length is not None and len_builtin(raw) != length:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_PIN", f"{label} length differs")
        if len_builtin(raw) > max_payload_bytes:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_TYPE", f"{label} exceeds byte bound")
        return raw

    def exact_tuple10(value: object, label: str) -> tuple[bytes, ...]:
        if type_builtin(value) is not tuple_type or len_builtin(value) != 10:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_TYPE",
                f"{label} must be one exact tuple of ten items",
            )
        items: list[bytes] = []
        for index, item in enumerate_builtin(value):
            items.append(exact_bytes(item, f"{label}[{index}]"))
        return tuple_builtin(items)

    def strict_ascii(raw: bytes, label: str) -> str:
        try:
            text = raw.decode("ascii", errors="strict")
        except unicode_decode_error_type as exc:
            raise error_type(
                f"FAIL_C1_PACKAGE0_MECHANICS_V1_PIN: {label} is not strict ASCII"
            ) from exc
        if not text:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_PIN", f"{label} is empty")
        return text

    def restore_numbers(value: object) -> object:
        if type_builtin(value) is float_type:
            if value.is_integer() and abs_builtin(value) <= safe_integer:
                return int_type(value)
            return value
        if type_builtin(value) is list_type:
            return [restore_numbers(item) for item in value]
        if type_builtin(value) is dict_type:
            return {key: restore_numbers(item) for key, item in value.items()}
        return value

    def decode(raw: bytes, label: str) -> object:
        try:
            decoded = decode_canonical(raw)
            restored = restore_numbers(decoded)
            if canonicalize(restored) != raw:
                fail(
                    "FAIL_C1_PACKAGE0_MECHANICS_V1_CANONICAL",
                    f"{label} differs after exact integer restoration",
                )
            return restored
        except error_type:
            raise
        except exception_type as exc:
            raise error_type(
                f"FAIL_C1_PACKAGE0_MECHANICS_V1_CANONICAL: {label} is not exact RFC8785"
            ) from exc

    def exact_object(
        value: object, fields: tuple[str, ...] | frozenset[str], label: str
    ) -> dict[str, object]:
        expected = frozenset_builtin(fields)
        if (
            type_builtin(value) is not dict_type
            or frozenset_builtin(value) != expected
        ):
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA",
                f"{label} exact field set differs",
            )
        return value

    def text(value: object, label: str) -> str:
        if type_builtin(value) is not str_type:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", f"{label} must be exact text")
        try:
            value.encode("ascii", errors="strict")
        except unicode_encode_error_type as exc:
            raise error_type(
                f"FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA: {label} is not ASCII"
            ) from exc
        if not value:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", f"{label} is empty")
        return value

    def integer(value: object, label: str, *, minimum: int = 0) -> int:
        if (
            type_builtin(value) is not int_type
            or value < minimum
            or value > safe_integer
        ):
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA",
                f"{label} exact integer differs",
            )
        return value

    def boolean(value: object, label: str) -> bool:
        if type_builtin(value) is not bool_type:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA",
                f"{label} exact boolean differs",
            )
        return value

    def hex64(value: object, label: str) -> str:
        item = text(value, label)
        if hex64_re.fullmatch(item) is None:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", f"{label} hex differs")
        return item

    def hex128(value: object, label: str) -> str:
        item = text(value, label)
        if hex128_re.fullmatch(item) is None:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SIGNATURE", f"{label} hex differs")
        return item

    def digest(value: object, label: str) -> str:
        item = text(value, label)
        if digest_re.fullmatch(item) is None:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", f"{label} digest differs")
        return item

    def identifier(value: object, label: str, *, prefix: str | None = None) -> str:
        item = text(value, label)
        if prefix is not None:
            if (
                not item.startswith(prefix)
                or hex64_re.fullmatch(item[len_builtin(prefix) :]) is None
            ):
                fail(
                    "FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA",
                    f"{label} content-ID prefix differs",
                )
        elif identifier_re.fullmatch(item) is None:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA",
                f"{label} content-ID syntax differs",
            )
        return item

    def string_array(value: object, label: str) -> tuple[str, ...]:
        if type_builtin(value) is not list_type:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", f"{label} must be an array")
        return tuple_builtin(
            text(item, f"{label}[{index}]")
            for index, item in enumerate_builtin(value)
        )

    def validate_wire_value(
        value: object,
        wire_type: object,
        label: str,
    ) -> None:
        profile = text(wire_type, f"{label} wire type")
        if profile == "EXACT_NONEMPTY_ASCII_ENUM_OR_IDENTIFIER_STRING":
            text(value, label)
            return
        if profile == "EXACT_ASCII_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX":
            identifier(value, label)
            return
        if profile == "EXACT_ASCII_SHA256_COLON_64_LOWERHEX":
            digest(value, label)
            return
        if profile == "EXACT_JSON_SAFE_INTEGER_MIN_0_BOOL_FORBIDDEN":
            integer(value, label)
            return
        if profile == "EXACT_JSON_BOOLEAN":
            boolean(value, label)
            return
        if profile == "EXACT_NONEMPTY_ASCII_SERVICE_IDENTIFIER":
            text(value, label)
            return
        if profile == (
            "EXACT_JSON_NULL_OR_EXTERNAL_ANCHOR_CONTENT_IDENTIFIER_"
            "PREFIX_PLUS_64_LOWERHEX"
        ):
            if value is not None:
                identifier(
                    value,
                    label,
                    prefix="phase2b_actual_unsealed_960_external_anchor_v1_",
                )
            return
        if profile in (
            "EXACT_JSON_ARRAY_OF_CONTENT_IDENTIFIERS",
            "EXACT_JSON_ARRAY_OF_SHA256_COLON_64_LOWERHEX",
        ):
            values = string_array(value, label)
            for index, item in enumerate_builtin(values):
                if profile == "EXACT_JSON_ARRAY_OF_CONTENT_IDENTIFIERS":
                    identifier(item, f"{label}[{index}]")
                else:
                    digest(item, f"{label}[{index}]")
            return
        if profile == "EXACT_JSON_ARRAY_OF_SAFE_INTEGERS_BOOL_FORBIDDEN":
            if type_builtin(value) is not list_type:
                fail(
                    "FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA",
                    f"{label} must be an array",
                )
            for index, item in enumerate_builtin(value):
                integer(item, f"{label}[{index}]")
            return
        fail(
            "FAIL_C1_PACKAGE0_MECHANICS_V1_INTERNAL",
            f"{label} has unsupported wire type {profile}",
        )

    def raw_digest(raw: bytes) -> str:
        return "sha256:" + sha256(raw).hexdigest()

    def body_content_id(descriptor: dict[str, object], body: dict[str, object]) -> str:
        domain = text(descriptor["content_id_domain_ascii"], "content-ID domain").encode("ascii")
        prefix = text(descriptor["content_id_prefix"], "content-ID prefix")
        return prefix + sha256(domain + canonicalize(body)).hexdigest()

    def exact_equal(left: object, right: object) -> bool:
        if type_builtin(left) is not type_builtin(right):
            return False
        if type_builtin(left) is dict_type:
            return frozenset_builtin(left) == frozenset_builtin(right) and all_builtin(
                exact_equal(left[key], right[key]) for key in left
            )
        if type_builtin(left) in (list_type, tuple_type):
            return len_builtin(left) == len_builtin(right) and all_builtin(
                exact_equal(a, b)
                for a, b in zip_builtin(left, right, strict=True)
            )
        return left == right

    def verify_ed25519(
        public_key_bytes: bytes,
        signature_hex: str,
        preimage: bytes,
        label: str,
    ) -> None:
        if public_key_type is None or invalid_signature_type is None:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_BACKEND",
                "Ed25519 verification backend is unavailable",
            )
        try:
            public_key_type.from_public_bytes(public_key_bytes).verify(
                bytes_type.fromhex(signature_hex), preimage
            )
        except (invalid_signature_type, type_error_type, value_error_type) as exc:
            raise error_type(
                f"FAIL_C1_PACKAGE0_MECHANICS_V1_SIGNATURE: {label} signature rejected"
            ) from exc

    def authorization_tuple(row: dict[str, object]) -> tuple[object, ...]:
        return (
            row["artifact_kind"],
            row["signature_ordinal"],
            row["role"],
            row["purpose"],
            row["key_epoch"],
            row["signature_domain_ascii"],
        )

    def generic_signature_preimage(
        descriptor: dict[str, object],
        *,
        artifact_schema_version: str,
        content_id: str,
        body: dict[str, object],
        slot: dict[str, object],
        signer_key_id: str,
    ) -> bytes:
        subject = {
            "schema_version": descriptor["signature_preimage_schema_version"],
            "artifact_kind": descriptor["signature_preimage_artifact_kind"],
            "artifact_schema_version": artifact_schema_version,
            "content_id": content_id,
            "body": body,
            "signature_ordinal": slot["ordinal"],
            "role": slot["role"],
            "purpose": slot["purpose"],
            "key_epoch": slot["key_epoch"],
            "signer_key_id": signer_key_id,
        }
        return text(slot["signature_domain_ascii"], "signature domain").encode(
            "ascii"
        ) + canonicalize(subject)

    def verify_generic_signature(
        signature_value: object,
        *,
        descriptor: dict[str, object],
        artifact_schema_version: str,
        content_id: str,
        body: dict[str, object],
        slot: dict[str, object],
        public_key_bytes: bytes,
        expected_key_id: str,
        expected_authorization_rows: tuple[tuple[object, ...], ...] | None,
        label: str,
    ) -> None:
        signature = exact_object(signature_value, detached_signature_fields, label)
        if (
            signature["schema_version"] != expected_detached_signature_schema
            or signature["algorithm"] != expected_signature_algorithm
            or signature["role"] != slot["role"]
            or signature["purpose"] != slot["purpose"]
            or integer(signature["key_epoch"], f"{label}.key_epoch", minimum=1)
            != slot["key_epoch"]
            or signature["signer_key_id"] != expected_key_id
        ):
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_SIGNATURE",
                f"{label} metadata differs",
            )
        if expected_authorization_rows is not None:
            expected_row = (
                descriptor["signature_preimage_artifact_kind"],
                slot["ordinal"],
                slot["role"],
                slot["purpose"],
                slot["key_epoch"],
                slot["signature_domain_ascii"],
            )
            if expected_authorization_rows.count(expected_row) != 1:
                fail(
                    "FAIL_C1_PACKAGE0_MECHANICS_V1_SIGNATURE",
                    f"{label} authorization row differs",
                )
        verify_ed25519(
            public_key_bytes,
            hex128(signature["signature_hex"], f"{label}.signature_hex"),
            generic_signature_preimage(
                descriptor,
                artifact_schema_version=artifact_schema_version,
                content_id=content_id,
                body=body,
                slot=slot,
                signer_key_id=expected_key_id,
            ),
            label,
        )

    def verify_catalog_and_pins(
        *,
        catalog_payload: bytes,
        out_of_band_expected_catalog_id_bytes: bytes,
        out_of_band_expected_catalog_payload_sha256_digest: bytes,
        out_of_band_expected_readiness_dag_id_bytes: bytes,
        out_of_band_expected_readiness_dag_payload_sha256_digest: bytes,
        out_of_band_expected_progression_plan_payload_sha256_digest: bytes,
        out_of_band_expected_gap_inventory_source_sha256_digest: bytes,
        out_of_band_expected_statistical_validation_source_sha256_digest: bytes,
    ) -> None:
        if catalog_payload != expected_catalog_payload:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_PIN",
                "catalog payload differs from the inspected exact bytes",
            )
        if (
            strict_ascii(
                out_of_band_expected_catalog_id_bytes,
                "out_of_band_expected_catalog_id_bytes",
            )
            != expected_catalog_id
            or out_of_band_expected_catalog_payload_sha256_digest
            != expected_catalog_sha
            or out_of_band_expected_readiness_dag_id_bytes
            != expected_readiness_id
            or out_of_band_expected_readiness_dag_payload_sha256_digest
            != expected_readiness_sha
            or out_of_band_expected_progression_plan_payload_sha256_digest
            != expected_plan_sha
            or out_of_band_expected_gap_inventory_source_sha256_digest
            != expected_gap_sha
            or out_of_band_expected_statistical_validation_source_sha256_digest
            != expected_statistical_sha
        ):
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_PIN",
                "one or more inspected catalog pins differ",
            )

    def verify_signer_registry(
        raw: bytes,
        *,
        expected_id_bytes: bytes,
        expected_payload_sha: bytes,
        expected_entries_root_sha: bytes,
        governance_root_key_id_bytes: bytes,
        governance_root_public_key_bytes: bytes,
        custodian_root_key_id_bytes: bytes,
        custodian_root_public_key_bytes: bytes,
    ) -> tuple[dict[str, dict[str, object]], str, str, str]:
        envelope_fields = tuple_builtin(
            signer_schema["envelope_fields_in_exact_order"]
        )
        body_fields = tuple_builtin(signer_schema["body_fields_in_exact_order"])
        envelope = exact_object(decode(raw, "signer registry"), envelope_fields, "signer registry")
        body = exact_object(envelope["body"], body_fields, "signer registry body")
        expected_schema_version = text(signer_schema["schema_version"], "registry schema")
        if (
            envelope["schema_version"] != expected_schema_version
            or body["schema_version"] != expected_schema_version
            or body["status"] != expected_registry_status
            or body["claim_level"] != expected_registry_claim_level
            or text(body["registry_epoch"], "registry epoch")
            != expected_registry_epoch
            or body["catalog_id"] != expected_catalog_id
        ):
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", "signer registry literals differ")
        entries = body["entries_in_exact_order"]
        rows = body["authorized_signature_rows_in_exact_order"]
        if type_builtin(entries) is not list_type or len_builtin(entries) != len_builtin(expected_registry_roles):
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", "registry entry cardinality differs")
        if type_builtin(rows) is not list_type:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", "authorization rows must be an array")
        row_fields = (
            "artifact_kind",
            "signature_ordinal",
            "role",
            "purpose",
            "key_epoch",
            "signature_domain_ascii",
        )
        parsed_rows: list[tuple[object, ...]] = []
        for index, row_value in enumerate_builtin(rows):
            row = exact_object(row_value, row_fields, f"authorization row {index}")
            integer(row["signature_ordinal"], f"authorization row {index} ordinal")
            integer(row["key_epoch"], f"authorization row {index} epoch", minimum=1)
            parsed_rows.append(authorization_tuple(row))
        if tuple_builtin(parsed_rows) != expected_authorization_rows:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", "authorization rows differ")
        entry_fields = (
            "entry_ordinal",
            "role",
            "control_domain_id",
            "key_id",
            "key_epoch",
            "algorithm",
            "public_key_hex",
            "authorized_signature_purposes_in_exact_order",
        )
        by_role: dict[str, dict[str, object]] = {}
        key_ids: list[str] = []
        control_domains: list[str] = []
        registry_public_keys: list[bytes] = []
        for index, (expected_role, entry_value) in enumerate_builtin(
            zip_builtin(expected_registry_roles, entries, strict=True)
        ):
            entry = exact_object(entry_value, entry_fields, f"registry entry {index}")
            key_id = identifier(entry["key_id"], f"registry entry {index} key ID")
            control_domain = identifier(
                entry["control_domain_id"],
                f"registry entry {index} control domain",
            )
            expected_purposes: list[str] = []
            for row in expected_authorization_rows:
                if row[2] == expected_role and row[3] not in expected_purposes:
                    expected_purposes.append(str_builtin(row[3]))
            if (
                text(entry["entry_ordinal"], f"registry entry {index} ordinal")
                != str_builtin(index)
                or entry["role"] != expected_role
                or integer(entry["key_epoch"], f"registry entry {index} epoch", minimum=1) != 1
                or entry["algorithm"] != expected_signature_algorithm
                or string_array(
                    entry["authorized_signature_purposes_in_exact_order"],
                    f"registry entry {index} purposes",
                )
                != tuple_builtin(expected_purposes)
            ):
                fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", f"registry entry {index} differs")
            public_key_hex = hex64(entry["public_key_hex"], f"registry entry {index} public key")
            by_role[str_builtin(expected_role)] = {
                **entry,
                "key_id": key_id,
                "control_domain_id": control_domain,
                "public_key_hex": public_key_hex,
            }
            key_ids.append(key_id)
            control_domains.append(control_domain)
            registry_public_keys.append(bytes_type.fromhex(public_key_hex))
        if (
            len_builtin(set_builtin(key_ids)) != len_builtin(key_ids)
            or len_builtin(set_builtin(control_domains)) != len_builtin(control_domains)
            or len_builtin(set_builtin(registry_public_keys))
            != len_builtin(registry_public_keys)
        ):
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", "registry key or control domains overlap")
        entry_root = "sha256:" + sha256(
            text(signer_schema["entry_root_domain_ascii"], "entry root domain").encode("ascii")
            + canonicalize(entries)
        ).hexdigest()
        authorization_root = "sha256:" + sha256(
            text(signer_schema["authorization_root_domain_ascii"], "authorization root domain").encode("ascii")
            + canonicalize(rows)
        ).hexdigest()
        if body["entries_root_sha256"] != entry_root or body["authorized_signature_rows_root_sha256"] != authorization_root:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_CONTENT_ID", "registry ordered root differs")
        registry_id = body_content_id(signer_schema, body)
        if envelope["signer_registry_id"] != registry_id:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_CONTENT_ID", "registry content ID differs")
        if strict_ascii(expected_id_bytes, "expected signer registry ID") != registry_id or expected_payload_sha != sha256(raw).digest() or expected_entries_root_sha != bytes_type.fromhex(entry_root[7:]):
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_PIN", "registry OOB pin differs")
        governance_root_key_id = strict_ascii(governance_root_key_id_bytes, "governance root key ID")
        custodian_root_key_id = strict_ascii(custodian_root_key_id_bytes, "custodian root key ID")
        governance_root_public_key = exact_bytes(
            governance_root_public_key_bytes,
            "governance root public key",
            length=32,
        )
        custodian_root_public_key = exact_bytes(
            custodian_root_public_key_bytes,
            "custodian root public key",
            length=32,
        )
        identifier(governance_root_key_id, "governance root key ID")
        identifier(custodian_root_key_id, "custodian root key ID")
        if (
            governance_root_key_id == custodian_root_key_id
            or governance_root_key_id in key_ids
            or custodian_root_key_id in key_ids
            or governance_root_public_key == custodian_root_public_key
            or governance_root_public_key in registry_public_keys
            or custodian_root_public_key in registry_public_keys
        ):
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_PIN", "root and registry key IDs overlap")
        signatures = envelope["signatures_in_exact_order"]
        slots = signer_schema["signature_slots_in_exact_order"]
        if type_builtin(signatures) is not list_type or len_builtin(signatures) != 2 or type_builtin(slots) is not list_type or len_builtin(slots) != 2:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SIGNATURE", "registry signature cardinality differs")
        for index, (signature, slot, root_key_id, public_key) in enumerate_builtin(
            (
                (signatures[0], slots[0], governance_root_key_id, governance_root_public_key),
                (signatures[1], slots[1], custodian_root_key_id, custodian_root_public_key),
            )
        ):
            verify_generic_signature(
                signature,
                descriptor=signer_schema,
                artifact_schema_version=expected_schema_version,
                content_id=registry_id,
                body=body,
                slot=slot,
                public_key_bytes=public_key,
                expected_key_id=root_key_id,
                expected_authorization_rows=None,
                label=f"registry root signature {index}",
            )
        return by_role, registry_id, raw_digest(raw), entry_root

    def content_id_from_parts(
        domain_ascii: object,
        prefix_value: object,
        body: dict[str, object],
        label: str,
    ) -> str:
        domain = text(domain_ascii, f"{label} content-ID domain").encode("ascii")
        prefix = text(prefix_value, f"{label} content-ID prefix")
        return prefix + sha256(domain + canonicalize(body)).hexdigest()

    def verify_registry_signature_slots(
        signatures_value: object,
        *,
        descriptor: dict[str, object],
        artifact_schema_version: str,
        content_id: str,
        body: dict[str, object],
        entries_by_role: dict[str, dict[str, object]],
        label: str,
    ) -> None:
        signatures = signatures_value
        slots = descriptor["signature_slots_in_exact_order"]
        if type_builtin(signatures) is not list_type or type_builtin(slots) is not list_type or len_builtin(signatures) != len_builtin(slots):
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SIGNATURE", f"{label} signature cardinality differs")
        for index, (signature, slot) in enumerate_builtin(
            zip_builtin(signatures, slots, strict=True)
        ):
            if type_builtin(slot) is not dict_type:
                fail("FAIL_C1_PACKAGE0_MECHANICS_V1_INTERNAL", f"{label} slot differs")
            role = text(slot["role"], f"{label} slot role")
            entry = entries_by_role.get(role)
            if entry is None:
                fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SIGNATURE", f"{label} role absent")
            verify_generic_signature(
                signature,
                descriptor=descriptor,
                artifact_schema_version=artifact_schema_version,
                content_id=content_id,
                body=body,
                slot=slot,
                public_key_bytes=bytes_type.fromhex(
                    str_builtin(entry["public_key_hex"])
                ),
                expected_key_id=str_builtin(entry["key_id"]),
                expected_authorization_rows=expected_authorization_rows,
                label=f"{label} signature {index}",
            )

    def verify_package_outer(
        raw: bytes,
        *,
        descriptor: dict[str, object],
        entries_by_role: dict[str, dict[str, object]],
        registry_id: str,
        registry_payload_sha: str,
        registry_entries_root: str,
        campaign_id: str,
        attempt_id: str,
        label: str,
    ) -> tuple[dict[str, object], dict[str, object], str, str]:
        envelope = exact_object(
            decode(raw, label),
            tuple_builtin(descriptor["envelope_fields_in_exact_order"]),
            label,
        )
        body = exact_object(
            envelope["body"],
            tuple_builtin(descriptor["body_fields_in_exact_order"]),
            f"{label} body",
        )
        schema_version = text(descriptor["schema_version"], f"{label} schema")
        required_literals = descriptor["body_required_literals"]
        if type_builtin(required_literals) is not dict_type:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_INTERNAL", f"{label} literals differ")
        if envelope["schema_version"] != schema_version or body["schema_version"] != schema_version:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", f"{label} schema differs")
        field_types = descriptor["body_field_types_in_exact_order"]
        if type_builtin(field_types) is not list_type or len_builtin(field_types) != len_builtin(body):
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_INTERNAL",
                f"{label} field-type manifest differs",
            )
        for field_index, field_row_value in enumerate_builtin(field_types):
            field_row = exact_object(
                field_row_value,
                ("field_name", "exact_wire_type"),
                f"{label} field manifest {field_index}",
            )
            field_name = text(
                field_row["field_name"],
                f"{label} field manifest {field_index} name",
            )
            validate_wire_value(
                body[field_name],
                field_row["exact_wire_type"],
                f"{label}.{field_name}",
            )
        for field_name, expected_value in required_literals.items():
            if not exact_equal(body[field_name], expected_value):
                fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", f"{label}.{field_name} literal differs")
        if (
            body["catalog_id"] != expected_catalog_id
            or body["signer_registry_id"] != registry_id
            or body["signer_registry_payload_sha256"] != registry_payload_sha
            or body["signer_registry_entries_root_sha256"] != registry_entries_root
            or body["campaign_id"] != campaign_id
            or body["attempt_id"] != attempt_id
        ):
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE", f"{label} lineage differs")
        predecessor_ids = string_array(
            body["predecessor_package_instance_ids_in_exact_order"],
            f"{label} predecessor IDs",
        )
        predecessor_shas = string_array(
            body["predecessor_package_instance_payload_sha256s_in_exact_order"],
            f"{label} predecessor SHAs",
        )
        predecessor_ordinals = tuple_builtin(
            descriptor["predecessor_package_ordinals_in_exact_order"]
        )
        if len_builtin(predecessor_ids) != len_builtin(predecessor_ordinals) or len_builtin(predecessor_shas) != len_builtin(predecessor_ordinals):
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE", f"{label} predecessor cardinality differs")
        for position, (predecessor_id, predecessor_sha, predecessor_ordinal) in enumerate_builtin(
            zip_builtin(predecessor_ids, predecessor_shas, predecessor_ordinals, strict=True)
        ):
            predecessor_descriptor = package_schemas[
                integer(predecessor_ordinal, f"{label} predecessor ordinal")
            ]
            identifier(
                predecessor_id,
                f"{label} predecessor ID {position}",
                prefix=str_builtin(predecessor_descriptor["content_id_prefix"]),
            )
            digest(predecessor_sha, f"{label} predecessor SHA {position}")
        package_id = content_id_from_parts(
            descriptor["content_id_domain_ascii"],
            descriptor["content_id_prefix"],
            body,
            label,
        )
        if envelope["package_id"] != package_id:
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_CONTENT_ID", f"{label} content ID differs")
        verify_registry_signature_slots(
            envelope["signatures_in_exact_order"],
            descriptor=descriptor,
            artifact_schema_version=schema_version,
            content_id=package_id,
            body=body,
            entries_by_role=entries_by_role,
            label=label,
        )
        return envelope, body, package_id, raw_digest(raw)

    def verify_shallow_p15_p16(
        raw_p15: bytes,
        raw_p16: bytes,
        *,
        entries_by_role: dict[str, dict[str, object]],
        registry_id: str,
        registry_payload_sha: str,
        registry_entries_root: str,
        campaign_id: str,
        attempt_id: str,
    ) -> tuple[
        tuple[dict[str, object], dict[str, object], str, str],
        tuple[dict[str, object], dict[str, object], str, str],
    ]:
        p15 = verify_package_outer(
            raw_p15,
            descriptor=package15_schema,
            entries_by_role=entries_by_role,
            registry_id=registry_id,
            registry_payload_sha=registry_payload_sha,
            registry_entries_root=registry_entries_root,
            campaign_id=campaign_id,
            attempt_id=attempt_id,
            label="package15 shallow outer",
        )
        p16 = verify_package_outer(
            raw_p16,
            descriptor=package16_schema,
            entries_by_role=entries_by_role,
            registry_id=registry_id,
            registry_payload_sha=registry_payload_sha,
            registry_entries_root=registry_entries_root,
            campaign_id=campaign_id,
            attempt_id=attempt_id,
            label="package16 shallow outer",
        )
        predecessor_ordinals = tuple_builtin(
            package16_schema["predecessor_package_ordinals_in_exact_order"]
        )
        try:
            p15_position = predecessor_ordinals.index(15)
        except value_error_type as exc:
            raise error_type(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_INTERNAL: package16 lacks package15 predecessor"
            ) from exc
        if (
            p16[1]["predecessor_package_instance_ids_in_exact_order"][p15_position]
            != p15[2]
            or p16[1]["predecessor_package_instance_payload_sha256s_in_exact_order"][p15_position]
            != p15[3]
        ):
            fail("FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE", "package16 does not shallow-bind package15")
        return p15, p16

    def verify_stage_events(
        raws: tuple[bytes, ...],
        *,
        entries_by_role: dict[str, dict[str, object]],
        registry_id: str,
        registry_payload_sha: str,
        registry_entries_root: str,
        campaign_id: str,
        attempt_id: str,
        p15: tuple[dict[str, object], dict[str, object], str, str],
        p16: tuple[dict[str, object], dict[str, object], str, str],
    ) -> tuple[tuple[dict[str, object], dict[str, object], str, str], ...]:
        verified: list[tuple[dict[str, object], dict[str, object], str, str]] = []
        gate_schema_by_id = {
            str_builtin(item["gate_id"]): item for item in gate_schemas
        }
        stage_index_by_name = {
            name: index for index, name in enumerate_builtin(stage_names)
        }
        for index, (raw, descriptor) in enumerate_builtin(
            zip_builtin(raws, stage_schemas, strict=True)
        ):
            label = f"stage event {index}"
            envelope = exact_object(
                decode(raw, label),
                tuple_builtin(descriptor["event_envelope_fields_in_exact_order"]),
                label,
            )
            body = exact_object(
                envelope["body"],
                tuple_builtin(descriptor["event_body_fields_in_exact_order"]),
                f"{label} body",
            )
            schema_version = text(descriptor["event_schema_version"], f"{label} schema")
            expected_stage = stage_names[index]
            expected_predecessor_names = tuple_builtin(
                descriptor["predecessor_nodes_in_exact_order"]
            )
            if (
                envelope["schema_version"] != schema_version
                or body["schema_version"] != schema_version
                or body["status"] != expected_stage_status
                or body["claim_level"] != expected_stage_claim_level
                or body["catalog_id"] != expected_catalog_id
                or body["signer_registry_id"] != registry_id
                or body["signer_registry_payload_sha256"] != registry_payload_sha
                or body["signer_registry_entries_root_sha256"] != registry_entries_root
                or body["campaign_id"] != campaign_id
                or body["attempt_id"] != attempt_id
                or integer(body["stage_ordinal"], f"{label} ordinal") != index
                or body["stage"] != expected_stage
                or integer(body["event_sequence"], f"{label} sequence") != index
                or boolean(body["stage_occurred"], f"{label} occurred")
                is not expected_stage_occurred
                or body["first_failure_code"] != expected_stage_first_failure
                or string_array(
                    body["predecessor_node_names_in_exact_order"],
                    f"{label} predecessor names",
                )
                != expected_predecessor_names
            ):
                fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", f"{label} required literal differs")
            if index < 9:
                if body["terminal_disposition_or_none"] is not None:
                    fail("FAIL_C1_PACKAGE0_MECHANICS_V1_TERMINAL", f"{label} disposition must be null")
            else:
                terminal_disposition = text(
                    body["terminal_disposition_or_none"],
                    "terminal disposition",
                )
                if terminal_disposition not in terminal_dispositions:
                    fail(
                        "FAIL_C1_PACKAGE0_MECHANICS_V1_TERMINAL",
                        "terminal disposition differs",
                    )
            predecessor_ids = string_array(
                body["predecessor_evidence_or_event_ids_in_exact_order"],
                f"{label} predecessor IDs",
            )
            predecessor_shas = string_array(
                body["predecessor_evidence_or_event_payload_sha256s_in_exact_order"],
                f"{label} predecessor SHAs",
            )
            if len_builtin(predecessor_ids) != len_builtin(expected_predecessor_names) or len_builtin(predecessor_shas) != len_builtin(expected_predecessor_names):
                fail("FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE", f"{label} predecessor cardinality differs")
            for position, (node_name, predecessor_id, predecessor_sha) in enumerate_builtin(
                zip_builtin(expected_predecessor_names, predecessor_ids, predecessor_shas, strict=True)
            ):
                if node_name in stage_index_by_name:
                    prior_index = stage_index_by_name[node_name]
                    if prior_index >= index or predecessor_id != verified[prior_index][2] or predecessor_sha != verified[prior_index][3]:
                        fail("FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE", f"{label} stage predecessor differs")
                else:
                    gate_descriptor = gate_schema_by_id.get(node_name)
                    if gate_descriptor is None:
                        fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", f"{label} predecessor node unknown")
                    identifier(
                        predecessor_id,
                        f"{label} gate predecessor ID {position}",
                        prefix=str_builtin(gate_descriptor["content_id_prefix"]),
                    )
                    digest(predecessor_sha, f"{label} gate predecessor SHA {position}")
            subject_kinds = string_array(
                body["subject_artifact_kinds_in_exact_order"],
                f"{label} subject kinds",
            )
            subject_ids = string_array(
                body["subject_artifact_ids_in_exact_order"],
                f"{label} subject IDs",
            )
            subject_shas = string_array(
                body["subject_artifact_payload_sha256s_in_exact_order"],
                f"{label} subject SHAs",
            )
            if not 1 <= len_builtin(subject_kinds) <= 64 or len_builtin(subject_ids) != len_builtin(subject_kinds) or len_builtin(subject_shas) != len_builtin(subject_kinds):
                fail("FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA", f"{label} subject cardinality differs")
            for position, (subject_kind, subject_id, subject_sha) in enumerate_builtin(
                zip_builtin(subject_kinds, subject_ids, subject_shas, strict=True)
            ):
                identifier(subject_id, f"{label} subject ID {position}")
                digest(subject_sha, f"{label} subject SHA {position}")
                if (
                    subject_kind in forbidden_stage_subject_kinds
                    or any_builtin(
                        subject_id.startswith(prefix)
                        for prefix in forbidden_stage_subject_prefixes
                    )
                    or (
                        index < 9
                        and (
                            subject_kind in terminal_subject_kinds
                            or any_builtin(
                                subject_id.startswith(prefix)
                                for prefix in terminal_subject_prefixes
                            )
                        )
                    )
                ):
                    fail(
                        "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE",
                        f"{label} has a forbidden forward subject reference",
                    )
            if index == 9 and (
                subject_kinds != terminal_subject_kinds
                or subject_ids != (p15[2], p16[2])
                or subject_shas != (p15[3], p16[3])
            ):
                fail("FAIL_C1_PACKAGE0_MECHANICS_V1_TERMINAL", "terminal P15/P16 binding differs")
            stage_id = content_id_from_parts(
                descriptor["content_id_domain_ascii"],
                descriptor["content_id_prefix"],
                body,
                label,
            )
            if envelope["stage_event_id"] != stage_id:
                fail("FAIL_C1_PACKAGE0_MECHANICS_V1_CONTENT_ID", f"{label} content ID differs")
            verify_registry_signature_slots(
                envelope["signatures_in_exact_order"],
                descriptor=descriptor,
                artifact_schema_version=schema_version,
                content_id=stage_id,
                body=body,
                entries_by_role=entries_by_role,
                label=label,
            )
            verified.append((envelope, body, stage_id, raw_digest(raw)))
        return tuple_builtin(verified)

    def verify_ledger_events(
        raws: tuple[bytes, ...],
        *,
        stages: tuple[
            tuple[dict[str, object], dict[str, object], str, str], ...
        ],
        entries_by_role: dict[str, dict[str, object]],
        registry_id: str,
        campaign_id: str,
        attempt_id: str,
    ) -> tuple[
        tuple[tuple[dict[str, object], dict[str, object], str, str], ...],
        str,
        str,
    ]:
        slots = ledger_contract["event_signature_slots_in_exact_stage_order"]
        if type_builtin(slots) is not list_type or len_builtin(slots) != 10:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_INTERNAL",
                "ledger signature-slot contract differs",
            )
        custodian = entries_by_role.get("CUSTODIAN")
        if custodian is None:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_SIGNATURE",
                "custodian signer is absent",
            )
        custodian_key_id = str_builtin(custodian["key_id"])
        custodian_public_key = bytes_type.fromhex(
            str_builtin(custodian["public_key_hex"])
        )
        verified: list[
            tuple[dict[str, object], dict[str, object], str, str]
        ] = []
        root_pairs: list[dict[str, str]] = []
        for index, (raw, stage, slot_value) in enumerate_builtin(
            zip_builtin(raws, stages, slots, strict=True)
        ):
            label = f"ledger event {index}"
            envelope = exact_object(
                decode(raw, label),
                ledger_envelope_fields,
                label,
            )
            body = exact_object(
                envelope["body"],
                ledger_body_fields,
                f"{label} body",
            )
            slot = exact_object(
                slot_value,
                ("ordinal", "role", "purpose", "key_epoch", "signature_domain_ascii"),
                f"{label} slot",
            )
            expected_predecessor = None if index == 0 else verified[index - 1][2]
            if (
                envelope["schema_version"] != ledger_envelope_schema
                or body["schema_version"] != ledger_body_schema
                or body["catalog_id"] != expected_catalog_id
                or body["campaign_id"] != campaign_id
                or body["attempt_id"] != attempt_id
                or integer(body["event_index"], f"{label} index") != index
                or body["stage"] != stage_names[index]
                or body["predecessor_event_id_or_none"] != expected_predecessor
                or body["payload_id"] != stage[2]
                or body["payload_sha256"] != stage[3]
                or body["signer_registry_id"] != registry_id
                or body["signer_key_id"] != custodian_key_id
            ):
                fail(
                    "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE",
                    f"{label} body projection differs",
                )
            if expected_predecessor is not None:
                identifier(
                    expected_predecessor,
                    f"{label} predecessor",
                    prefix=ledger_id_prefix,
                )
            identifier(body["payload_id"], f"{label} payload ID")
            digest(body["payload_sha256"], f"{label} payload SHA")
            signatures = envelope["signatures_in_exact_order"]
            if type_builtin(signatures) is not list_type or len_builtin(signatures) != 1:
                fail(
                    "FAIL_C1_PACKAGE0_MECHANICS_V1_SIGNATURE",
                    f"{label} signature cardinality differs",
                )
            signature = exact_object(
                signatures[0],
                detached_signature_fields,
                f"{label} signature",
            )
            artifact_kind = f"signed_custodian_stage_event_{index}_v1"
            expected_authorization = (
                artifact_kind,
                0,
                "CUSTODIAN",
                slot["purpose"],
                1,
                slot["signature_domain_ascii"],
            )
            if expected_authorization_rows.count(expected_authorization) != 1:
                fail(
                    "FAIL_C1_PACKAGE0_MECHANICS_V1_SIGNATURE",
                    f"{label} authorization row differs",
                )
            if (
                signature["schema_version"] != expected_detached_signature_schema
                or signature["algorithm"] != expected_signature_algorithm
                or signature["role"] != "CUSTODIAN"
                or signature["purpose"] != slot["purpose"]
                or integer(
                    signature["key_epoch"],
                    f"{label} signature epoch",
                    minimum=1,
                )
                != 1
                or signature["signer_key_id"] != custodian_key_id
                or integer(slot["ordinal"], f"{label} slot ordinal") != 0
                or slot["role"] != "CUSTODIAN"
                or integer(slot["key_epoch"], f"{label} slot epoch", minimum=1)
                != 1
            ):
                fail(
                    "FAIL_C1_PACKAGE0_MECHANICS_V1_SIGNATURE",
                    f"{label} signature metadata differs",
                )
            signature_subject = {
                "schema_version": ledger_signature_preimage_schema,
                "artifact_kind": artifact_kind,
                "artifact_schema_version": ledger_envelope_schema,
                "algorithm": expected_signature_algorithm,
                "body": body,
                "signature_ordinal": 0,
                "role": "CUSTODIAN",
                "purpose": slot["purpose"],
                "key_epoch": 1,
                "signer_key_id": custodian_key_id,
            }
            verify_ed25519(
                custodian_public_key,
                hex128(
                    signature["signature_hex"],
                    f"{label} signature hex",
                ),
                text(
                    slot["signature_domain_ascii"],
                    f"{label} signature domain",
                ).encode("ascii")
                + canonicalize(signature_subject),
                label,
            )
            event_subject = {
                "schema_version": ledger_envelope_schema,
                "body": body,
                "signatures_in_exact_order": signatures,
            }
            event_id = ledger_id_prefix + sha256(
                ledger_id_domain + canonicalize(event_subject)
            ).hexdigest()
            if envelope["signed_stage_event_id"] != event_id:
                fail(
                    "FAIL_C1_PACKAGE0_MECHANICS_V1_CONTENT_ID",
                    f"{label} content ID differs",
                )
            event_sha = raw_digest(raw)
            verified.append((envelope, body, event_id, event_sha))
            root_pairs.append(
                {
                    "signed_stage_event_id": event_id,
                    "signed_stage_event_payload_sha256": event_sha,
                }
            )
        event_ids = tuple_builtin(item[2] for item in verified)
        if len_builtin(set_builtin(event_ids)) != 10:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE",
                "ledger event IDs are not pairwise distinct",
            )
        chain_root = "sha256:" + sha256(
            ledger_chain_root_domain + canonicalize(root_pairs)
        ).hexdigest()
        chain_id = ledger_chain_id_prefix + chain_root[7:]
        return tuple_builtin(verified), chain_root, chain_id

    def verify_referenced_artifact(
        raw: bytes,
        *,
        contract: dict[str, object],
        entries_by_role: dict[str, dict[str, object]],
        campaign_id: str,
        attempt_id: str,
    ) -> tuple[dict[str, object], dict[str, object], str, str]:
        record_name = text(contract["record_name"], "referenced record name")
        label = f"referenced {record_name}"
        envelope = exact_object(
            decode(raw, label),
            tuple_builtin(contract["record_envelope_fields_in_exact_order"]),
            label,
        )
        body = exact_object(
            envelope["body"],
            tuple_builtin(contract["record_body_fields_in_exact_order"]),
            f"{label} body",
        )
        schema_version = text(contract["record_schema_version"], f"{label} schema")
        if (
            envelope["schema_version"] != schema_version
            or body["schema_version"] != schema_version
            or body["catalog_id"] != expected_catalog_id
            or body["campaign_id"] != campaign_id
            or body["attempt_id"] != attempt_id
        ):
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE",
                f"{label} required lineage differs",
            )
        field_types = contract["record_body_field_types_in_exact_order"]
        if type_builtin(field_types) is not list_type or len_builtin(field_types) != len_builtin(body):
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_INTERNAL",
                f"{label} field-type manifest differs",
            )
        for field_index, row_value in enumerate_builtin(field_types):
            row = exact_object(
                row_value,
                ("field_name", "exact_wire_type"),
                f"{label} field manifest {field_index}",
            )
            field_name = text(
                row["field_name"],
                f"{label} field manifest {field_index} name",
            )
            validate_wire_value(
                body[field_name],
                row["exact_wire_type"],
                f"{label}.{field_name}",
            )
        artifact_id = content_id_from_parts(
            contract["record_content_id_domain_ascii"],
            contract["record_content_id_prefix"],
            body,
            label,
        )
        if envelope["artifact_id"] != artifact_id:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_CONTENT_ID",
                f"{label} content ID differs",
            )
        signature_descriptor = {
            "signature_preimage_schema_version": contract[
                "signature_preimage_schema_version"
            ],
            "signature_preimage_artifact_kind": contract[
                "signature_preimage_artifact_kind"
            ],
            "signature_slots_in_exact_order": contract[
                "signature_slots_in_exact_order"
            ],
        }
        verify_registry_signature_slots(
            envelope["signatures_in_exact_order"],
            descriptor=signature_descriptor,
            artifact_schema_version=schema_version,
            content_id=artifact_id,
            body=body,
            entries_by_role=entries_by_role,
            label=label,
        )
        return envelope, body, artifact_id, raw_digest(raw)

    def verify_referenced_graph(
        raw_store: bytes,
        raw_time: bytes,
        raw_governance: bytes,
        raw_custodian: bytes,
        raw_anchor: bytes,
        *,
        entries_by_role: dict[str, dict[str, object]],
        campaign_id: str,
        attempt_id: str,
        ledger_events: tuple[
            tuple[dict[str, object], dict[str, object], str, str], ...
        ],
        ledger_raws: tuple[bytes, ...],
        chain_root: str,
        chain_id: str,
    ) -> tuple[
        tuple[
            tuple[dict[str, object], dict[str, object], str, str], ...
        ],
        dict[str, tuple[dict[str, object], dict[str, object], str, str]],
    ]:
        raw_values = (
            raw_store,
            raw_time,
            raw_governance,
            raw_custodian,
            raw_anchor,
        )
        verified = tuple_builtin(
            verify_referenced_artifact(
                raw,
                contract=contract,
                entries_by_role=entries_by_role,
                campaign_id=campaign_id,
                attempt_id=attempt_id,
            )
            for raw, contract in zip_builtin(
                raw_values,
                referenced_contracts,
                strict=True,
            )
        )
        by_name = {
            str_builtin(contract["record_name"]): item
            for contract, item in zip_builtin(
                referenced_contracts,
                verified,
                strict=True,
            )
        }
        store = by_name["store_if_absent_receipt"]
        time = by_name["logical_time_token"]
        governance = by_name["governance_checkpoint"]
        custodian = by_name["custodian_checkpoint"]
        anchor = by_name["external_anchor"]
        terminal = ledger_events[9]
        if (
            store[1]["transition_from"] != "ABSENT"
            or store[1]["transition_to"] != "PRESENT"
            or store[1]["mode"] != "IF_ABSENT"
            or store[1]["stored_artifact_id"] != terminal[2]
            or store[1]["stored_artifact_payload_sha256"] != terminal[3]
            or integer(
                store[1]["stored_artifact_size_bytes"],
                "store receipt artifact size",
            )
            != len_builtin(ledger_raws[9])
            or boolean(store[1]["append_applied"], "store receipt append")
            is not True
            or boolean(store[1]["readback_verified"], "store receipt readback")
            is not True
        ):
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_REFERENCED_GRAPH",
                "terminal store receipt projection differs",
            )
        previous_time = integer(
            time[1]["previous_logical_time"],
            "logical time previous",
        )
        current_time = integer(
            time[1]["current_logical_time"],
            "logical time current",
        )
        if (
            current_time != previous_time + 1
            or time[1]["subject_id"] != terminal[2]
            or time[1]["subject_payload_sha256"] != terminal[3]
        ):
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_REFERENCED_GRAPH",
                "logical-time projection differs",
            )
        checkpoint_projection = {
            "catalog_id": expected_catalog_id,
            "campaign_id": campaign_id,
            "attempt_id": attempt_id,
            "chain_id": chain_id,
            "chain_root_sha256": chain_root,
            "store_receipt_id": store[2],
            "store_receipt_payload_sha256": store[3],
            "time_token_id": time[2],
            "time_token_payload_sha256": time[3],
            "terminal_signed_ledger_event_id": terminal[2],
            "terminal_signed_ledger_event_payload_sha256": terminal[3],
        }
        for checkpoint_name, checkpoint in (
            ("governance", governance),
            ("custodian", custodian),
        ):
            for field_name, expected_value in checkpoint_projection.items():
                if not exact_equal(checkpoint[1][field_name], expected_value):
                    fail(
                        "FAIL_C1_PACKAGE0_MECHANICS_V1_REFERENCED_GRAPH",
                        f"{checkpoint_name} checkpoint projection differs",
                    )
        if (
            anchor[1]["chain_id"] != chain_id
            or anchor[1]["chain_root_sha256"] != chain_root
            or anchor[1]["governance_checkpoint_id"] != governance[2]
            or anchor[1]["governance_checkpoint_payload_sha256"] != governance[3]
            or anchor[1]["custodian_checkpoint_id"] != custodian[2]
            or anchor[1]["custodian_checkpoint_payload_sha256"] != custodian[3]
            or anchor[1]["previous_external_anchor_id_or_none"] is not None
            or integer(anchor[1]["anchor_sequence"], "anchor sequence") != 0
            or anchor[1]["effective_state"] != "ACTIVE"
            or boolean(anchor[1]["superseded"], "anchor superseded") is not False
            or boolean(anchor[1]["revoked"], "anchor revoked") is not False
        ):
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_REFERENCED_GRAPH",
                "external-anchor projection differs",
            )
        identifier(
            chain_id,
            "chain ID",
            prefix=ledger_chain_id_prefix,
        )
        if chain_id != ledger_chain_id_prefix + chain_root[7:]:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_REFERENCED_GRAPH",
                "local mechanics chain-ID formula differs",
            )
        if len_builtin(set_builtin(item[2] for item in verified)) != 5:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_CONTENT_ID",
                "referenced artifact IDs are not pairwise distinct",
            )
        return verified, by_name

    def verify_package0(
        raw: bytes,
        *,
        entries_by_role: dict[str, dict[str, object]],
        registry_id: str,
        registry_payload_sha: str,
        registry_entries_root: str,
        campaign_id: str,
        attempt_id: str,
        ledger_events: tuple[
            tuple[dict[str, object], dict[str, object], str, str], ...
        ],
        referenced_by_name: dict[
            str,
            tuple[dict[str, object], dict[str, object], str, str],
        ],
        chain_root: str,
    ) -> tuple[dict[str, object], dict[str, object], str, str]:
        package0 = verify_package_outer(
            raw,
            descriptor=package0_schema,
            entries_by_role=entries_by_role,
            registry_id=registry_id,
            registry_payload_sha=registry_payload_sha,
            registry_entries_root=registry_entries_root,
            campaign_id=campaign_id,
            attempt_id=attempt_id,
            label="package0",
        )
        body = package0[1]
        if (
            string_array(
                body["predecessor_package_instance_ids_in_exact_order"],
                "package0 predecessor IDs",
            )
            or string_array(
                body[
                    "predecessor_package_instance_payload_sha256s_in_exact_order"
                ],
                "package0 predecessor SHAs",
            )
        ):
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE",
                "package0 must have no predecessor package",
            )
        ledger_ids = tuple_builtin(item[2] for item in ledger_events)
        ledger_shas = tuple_builtin(item[3] for item in ledger_events)
        if (
            string_array(
                body["signed_stage_event_ids_in_exact_order"],
                "package0 ledger IDs",
            )
            != ledger_ids
            or string_array(
                body["signed_stage_event_payload_sha256s_in_exact_order"],
                "package0 ledger SHAs",
            )
            != ledger_shas
            or body["chain_root_sha256"] != chain_root
        ):
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE",
                "package0 ledger projection differs",
            )
        for contract in referenced_contracts:
            record_name = str_builtin(contract["record_name"])
            reference_fields = tuple_builtin(
                contract["outer_body_reference_fields_in_exact_order"]
            )
            item = referenced_by_name[record_name]
            if (
                body[reference_fields[0]] != item[2]
                or body[reference_fields[1]] != item[3]
            ):
                fail(
                    "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE",
                    f"package0 {record_name} reference differs",
                )
        return package0

    def issue_result(
        *,
        package0: tuple[dict[str, object], dict[str, object], str, str],
        registry_id: str,
        registry_payload_sha: str,
        registry_entries_root: str,
        campaign_id: str,
        attempt_id: str,
        chain_root: str,
        ledger_events: tuple[
            tuple[dict[str, object], dict[str, object], str, str], ...
        ],
    ) -> VerifiedPhase2BActualUnsealed960Package0MechanicsV1:
        values: dict[str, object] = {
            "schema_version": expected_schema_version,
            "status": expected_status,
            "claim_level": expected_claim_level,
            "disposition": disposition_type(
                "VERIFIED_PRESENTED_PACKAGE0_MECHANICS_NOT_R10"
            ),
            "reason": reason_type("EXACT_PACKAGE0_RAW_GRAPH_VERIFIED"),
            "catalog_id": expected_catalog_id,
            "catalog_payload_sha256": "sha256:" + expected_catalog_sha_text,
            "package_schema_id": str_builtin(package0_schema["package_schema_id"]),
            "package_id": package0[2],
            "package_payload_sha256": package0[3],
            "signer_registry_id": registry_id,
            "signer_registry_payload_sha256": registry_payload_sha,
            "signer_registry_entries_root_sha256": registry_entries_root,
            "campaign_id": campaign_id,
            "attempt_id": attempt_id,
            "chain_root_sha256": chain_root,
            "terminal_signed_stage_event_id": ledger_events[9][2],
            "terminal_signed_stage_event_payload_sha256": ledger_events[9][3],
            "ledger_event_count": 10,
            "stage_subject_event_count": 10,
            "referenced_artifact_count": 5,
            "total_signature_count": expected_total_signature_count,
        }
        values.update({name: True for name in mechanics_fields})
        values.update({name: False for name in nonclaim_fields})
        if tuple_builtin(values) != expected_result_fields:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_INTERNAL",
                "private result field order differs",
            )
        result = object_type.__new__(result_type)
        for field_name in expected_result_fields:
            object_type.__setattr__(result, field_name, values[field_name])
        snapshot = tuple_builtin(
            getattr_builtin(result, field_name)
            for field_name in expected_result_fields
        )
        registry[id_builtin(result)] = (result, snapshot)
        return result

    def require_result(
        candidate: object,
    ) -> VerifiedPhase2BActualUnsealed960Package0MechanicsV1:
        if type_builtin(candidate) is not result_type:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE",
                "candidate has no private Package0 result lineage",
            )
        record = registry.get(id_builtin(candidate))
        if record is None or record[0] is not candidate:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE",
                "candidate was not issued by this verifier",
            )
        current = tuple_builtin(
            getattr_builtin(candidate, field_name)
            for field_name in expected_result_fields
        )
        if not exact_equal(current, record[1]):
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE",
                "issued Package0 result was mutated",
            )
        return candidate

    def verify_presented(
        catalog_payload: bytes,
        raw_signer_registry_payload: bytes,
        raw_package0_envelope_payload: bytes,
        raw_package0_ledger_event_payloads_in_exact_order: tuple[bytes, ...],
        raw_package0_stage_subject_event_payloads_in_exact_order: tuple[bytes, ...],
        raw_store_if_absent_receipt_payload: bytes,
        raw_logical_time_token_payload: bytes,
        raw_governance_checkpoint_payload: bytes,
        raw_custodian_checkpoint_payload: bytes,
        raw_external_anchor_payload: bytes,
        raw_package15_envelope_payload: bytes,
        raw_package16_envelope_payload: bytes,
        *,
        out_of_band_expected_catalog_id_bytes: bytes,
        out_of_band_expected_catalog_payload_sha256_digest: bytes,
        out_of_band_expected_signer_registry_id_bytes: bytes,
        out_of_band_expected_signer_registry_payload_sha256_digest: bytes,
        out_of_band_expected_signer_registry_entries_root_sha256_digest: bytes,
        out_of_band_expected_governance_root_key_id_bytes: bytes,
        out_of_band_expected_governance_root_public_key_bytes: bytes,
        out_of_band_expected_custodian_root_key_id_bytes: bytes,
        out_of_band_expected_custodian_root_public_key_bytes: bytes,
        out_of_band_expected_readiness_dag_id_bytes: bytes,
        out_of_band_expected_readiness_dag_payload_sha256_digest: bytes,
        out_of_band_expected_progression_plan_payload_sha256_digest: bytes,
        out_of_band_expected_gap_inventory_source_sha256_digest: bytes,
        out_of_band_expected_statistical_validation_source_sha256_digest: bytes,
        expected_campaign_id_bytes: bytes,
        expected_attempt_id_bytes: bytes,
    ) -> VerifiedPhase2BActualUnsealed960Package0MechanicsV1:
        raw_catalog = exact_bytes(catalog_payload, "catalog_payload")
        raw_registry = exact_bytes(
            raw_signer_registry_payload,
            "raw_signer_registry_payload",
        )
        raw_package0 = exact_bytes(
            raw_package0_envelope_payload,
            "raw_package0_envelope_payload",
        )
        ledger_raws = exact_tuple10(
            raw_package0_ledger_event_payloads_in_exact_order,
            "raw_package0_ledger_event_payloads_in_exact_order",
        )
        stage_raws = exact_tuple10(
            raw_package0_stage_subject_event_payloads_in_exact_order,
            "raw_package0_stage_subject_event_payloads_in_exact_order",
        )
        raw_store = exact_bytes(
            raw_store_if_absent_receipt_payload,
            "raw_store_if_absent_receipt_payload",
        )
        raw_time = exact_bytes(
            raw_logical_time_token_payload,
            "raw_logical_time_token_payload",
        )
        raw_governance = exact_bytes(
            raw_governance_checkpoint_payload,
            "raw_governance_checkpoint_payload",
        )
        raw_custodian = exact_bytes(
            raw_custodian_checkpoint_payload,
            "raw_custodian_checkpoint_payload",
        )
        raw_anchor = exact_bytes(
            raw_external_anchor_payload,
            "raw_external_anchor_payload",
        )
        raw_p15 = exact_bytes(
            raw_package15_envelope_payload,
            "raw_package15_envelope_payload",
        )
        raw_p16 = exact_bytes(
            raw_package16_envelope_payload,
            "raw_package16_envelope_payload",
        )
        expected_catalog_id_bytes = exact_bytes(
            out_of_band_expected_catalog_id_bytes,
            "out_of_band_expected_catalog_id_bytes",
        )
        expected_catalog_digest = exact_bytes(
            out_of_band_expected_catalog_payload_sha256_digest,
            "out_of_band_expected_catalog_payload_sha256_digest",
            length=32,
        )
        expected_registry_id_bytes = exact_bytes(
            out_of_band_expected_signer_registry_id_bytes,
            "out_of_band_expected_signer_registry_id_bytes",
        )
        expected_registry_digest = exact_bytes(
            out_of_band_expected_signer_registry_payload_sha256_digest,
            "out_of_band_expected_signer_registry_payload_sha256_digest",
            length=32,
        )
        expected_entries_root_digest = exact_bytes(
            out_of_band_expected_signer_registry_entries_root_sha256_digest,
            "out_of_band_expected_signer_registry_entries_root_sha256_digest",
            length=32,
        )
        governance_root_key_id = exact_bytes(
            out_of_band_expected_governance_root_key_id_bytes,
            "out_of_band_expected_governance_root_key_id_bytes",
        )
        governance_root_public_key = exact_bytes(
            out_of_band_expected_governance_root_public_key_bytes,
            "out_of_band_expected_governance_root_public_key_bytes",
            length=32,
        )
        custodian_root_key_id = exact_bytes(
            out_of_band_expected_custodian_root_key_id_bytes,
            "out_of_band_expected_custodian_root_key_id_bytes",
        )
        custodian_root_public_key = exact_bytes(
            out_of_band_expected_custodian_root_public_key_bytes,
            "out_of_band_expected_custodian_root_public_key_bytes",
            length=32,
        )
        readiness_id = exact_bytes(
            out_of_band_expected_readiness_dag_id_bytes,
            "out_of_band_expected_readiness_dag_id_bytes",
        )
        readiness_digest = exact_bytes(
            out_of_band_expected_readiness_dag_payload_sha256_digest,
            "out_of_band_expected_readiness_dag_payload_sha256_digest",
            length=32,
        )
        plan_digest = exact_bytes(
            out_of_band_expected_progression_plan_payload_sha256_digest,
            "out_of_band_expected_progression_plan_payload_sha256_digest",
            length=32,
        )
        gap_digest = exact_bytes(
            out_of_band_expected_gap_inventory_source_sha256_digest,
            "out_of_band_expected_gap_inventory_source_sha256_digest",
            length=32,
        )
        statistical_digest = exact_bytes(
            out_of_band_expected_statistical_validation_source_sha256_digest,
            "out_of_band_expected_statistical_validation_source_sha256_digest",
            length=32,
        )
        campaign_id_raw = exact_bytes(
            expected_campaign_id_bytes,
            "expected_campaign_id_bytes",
        )
        attempt_id_raw = exact_bytes(
            expected_attempt_id_bytes,
            "expected_attempt_id_bytes",
        )
        campaign_id = strict_ascii(campaign_id_raw, "expected campaign ID")
        attempt_id = strict_ascii(attempt_id_raw, "expected attempt ID")
        if campaign_re.fullmatch(campaign_id) is None or attempt_re.fullmatch(attempt_id) is None:
            fail(
                "FAIL_C1_PACKAGE0_MECHANICS_V1_PIN",
                "campaign or attempt ID syntax differs",
            )
        verify_catalog_and_pins(
            catalog_payload=raw_catalog,
            out_of_band_expected_catalog_id_bytes=expected_catalog_id_bytes,
            out_of_band_expected_catalog_payload_sha256_digest=expected_catalog_digest,
            out_of_band_expected_readiness_dag_id_bytes=readiness_id,
            out_of_band_expected_readiness_dag_payload_sha256_digest=readiness_digest,
            out_of_band_expected_progression_plan_payload_sha256_digest=plan_digest,
            out_of_band_expected_gap_inventory_source_sha256_digest=gap_digest,
            out_of_band_expected_statistical_validation_source_sha256_digest=statistical_digest,
        )
        entries, registry_id, registry_sha, entries_root = verify_signer_registry(
            raw_registry,
            expected_id_bytes=expected_registry_id_bytes,
            expected_payload_sha=expected_registry_digest,
            expected_entries_root_sha=expected_entries_root_digest,
            governance_root_key_id_bytes=governance_root_key_id,
            governance_root_public_key_bytes=governance_root_public_key,
            custodian_root_key_id_bytes=custodian_root_key_id,
            custodian_root_public_key_bytes=custodian_root_public_key,
        )
        p15, p16 = verify_shallow_p15_p16(
            raw_p15,
            raw_p16,
            entries_by_role=entries,
            registry_id=registry_id,
            registry_payload_sha=registry_sha,
            registry_entries_root=entries_root,
            campaign_id=campaign_id,
            attempt_id=attempt_id,
        )
        stages = verify_stage_events(
            stage_raws,
            entries_by_role=entries,
            registry_id=registry_id,
            registry_payload_sha=registry_sha,
            registry_entries_root=entries_root,
            campaign_id=campaign_id,
            attempt_id=attempt_id,
            p15=p15,
            p16=p16,
        )
        ledger_events, chain_root, chain_id = verify_ledger_events(
            ledger_raws,
            stages=stages,
            entries_by_role=entries,
            registry_id=registry_id,
            campaign_id=campaign_id,
            attempt_id=attempt_id,
        )
        _, referenced_by_name = verify_referenced_graph(
            raw_store,
            raw_time,
            raw_governance,
            raw_custodian,
            raw_anchor,
            entries_by_role=entries,
            campaign_id=campaign_id,
            attempt_id=attempt_id,
            ledger_events=ledger_events,
            ledger_raws=ledger_raws,
            chain_root=chain_root,
            chain_id=chain_id,
        )
        package0 = verify_package0(
            raw_package0,
            entries_by_role=entries,
            registry_id=registry_id,
            registry_payload_sha=registry_sha,
            registry_entries_root=entries_root,
            campaign_id=campaign_id,
            attempt_id=attempt_id,
            ledger_events=ledger_events,
            referenced_by_name=referenced_by_name,
            chain_root=chain_root,
        )
        return issue_result(
            package0=package0,
            registry_id=registry_id,
            registry_payload_sha=registry_sha,
            registry_entries_root=entries_root,
            campaign_id=campaign_id,
            attempt_id=attempt_id,
            chain_root=chain_root,
            ledger_events=ledger_events,
        )

    def verify_real(
        catalog_payload: bytes,
        raw_signer_registry_payload: bytes,
        raw_package0_envelope_payload: bytes,
        raw_package0_ledger_event_payloads_in_exact_order: tuple[bytes, ...],
        raw_package0_stage_subject_event_payloads_in_exact_order: tuple[bytes, ...],
        raw_store_if_absent_receipt_payload: bytes,
        raw_logical_time_token_payload: bytes,
        raw_governance_checkpoint_payload: bytes,
        raw_custodian_checkpoint_payload: bytes,
        raw_external_anchor_payload: bytes,
        raw_package15_envelope_payload: bytes,
        raw_package16_envelope_payload: bytes,
        *,
        out_of_band_expected_catalog_id_bytes: bytes,
        out_of_band_expected_catalog_payload_sha256_digest: bytes,
        out_of_band_expected_signer_registry_id_bytes: bytes,
        out_of_band_expected_signer_registry_payload_sha256_digest: bytes,
        out_of_band_expected_signer_registry_entries_root_sha256_digest: bytes,
        out_of_band_expected_governance_root_key_id_bytes: bytes,
        out_of_band_expected_governance_root_public_key_bytes: bytes,
        out_of_band_expected_custodian_root_key_id_bytes: bytes,
        out_of_band_expected_custodian_root_public_key_bytes: bytes,
        out_of_band_expected_readiness_dag_id_bytes: bytes,
        out_of_band_expected_readiness_dag_payload_sha256_digest: bytes,
        out_of_band_expected_progression_plan_payload_sha256_digest: bytes,
        out_of_band_expected_gap_inventory_source_sha256_digest: bytes,
        out_of_band_expected_statistical_validation_source_sha256_digest: bytes,
        expected_campaign_id_bytes: bytes,
        expected_attempt_id_bytes: bytes,
    ) -> NoReturn:
        del (
            catalog_payload,
            raw_signer_registry_payload,
            raw_package0_envelope_payload,
            raw_package0_ledger_event_payloads_in_exact_order,
            raw_package0_stage_subject_event_payloads_in_exact_order,
            raw_store_if_absent_receipt_payload,
            raw_logical_time_token_payload,
            raw_governance_checkpoint_payload,
            raw_custodian_checkpoint_payload,
            raw_external_anchor_payload,
            raw_package15_envelope_payload,
            raw_package16_envelope_payload,
            out_of_band_expected_catalog_id_bytes,
            out_of_band_expected_catalog_payload_sha256_digest,
            out_of_band_expected_signer_registry_id_bytes,
            out_of_band_expected_signer_registry_payload_sha256_digest,
            out_of_band_expected_signer_registry_entries_root_sha256_digest,
            out_of_band_expected_governance_root_key_id_bytes,
            out_of_band_expected_governance_root_public_key_bytes,
            out_of_band_expected_custodian_root_key_id_bytes,
            out_of_band_expected_custodian_root_public_key_bytes,
            out_of_band_expected_readiness_dag_id_bytes,
            out_of_band_expected_readiness_dag_payload_sha256_digest,
            out_of_band_expected_progression_plan_payload_sha256_digest,
            out_of_band_expected_gap_inventory_source_sha256_digest,
            out_of_band_expected_statistical_validation_source_sha256_digest,
            expected_campaign_id_bytes,
            expected_attempt_id_bytes,
        )
        fail(
            "FAIL_C1_PACKAGE0_MECHANICS_V1_NOT_IMPLEMENTED",
            "real Package0 verification requires external operational evidence",
        )

    verify_presented.__name__ = (
        "verify_phase2b_actual_unsealed_960_durable_signed_custodian_"
        "ledger_and_signer_chain_mechanics_v1"
    )
    verify_real.__name__ = (
        "verify_real_phase2b_actual_unsealed_960_durable_signed_custodian_"
        "ledger_and_signer_chain_v1"
    )
    require_result.__name__ = (
        "require_verified_phase2b_actual_unsealed_960_durable_signed_"
        "custodian_ledger_and_signer_chain_mechanics_v1"
    )
    return verify_presented, require_result, verify_real


(
    verify_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1,
    require_verified_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1,
    verify_real_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1,
) = _build_private_boundary_v1()


def _assert_public_abi_v1() -> None:
    expected_positional_names = tuple(
        name for name, _ in POSITIONAL_PARAMETERS_IN_EXACT_ORDER_V1
    )
    expected_keyword_names = tuple(KEYWORD_ONLY_PARAMETERS_IN_EXACT_ORDER_V1)
    for function in (
        verify_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1,
        verify_real_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1,
    ):
        code = function.__code__
        if (
            code.co_posonlyargcount != 0
            or code.co_argcount != len(POSITIONAL_PARAMETERS_IN_EXACT_ORDER_V1)
            or code.co_kwonlyargcount != len(KEYWORD_ONLY_PARAMETERS_IN_EXACT_ORDER_V1)
            or tuple(
                code.co_varnames[: code.co_argcount]
            )
            != expected_positional_names
            or tuple(
                code.co_varnames[
                    code.co_argcount : code.co_argcount + code.co_kwonlyargcount
                ]
            )
            != expected_keyword_names
            or code.co_flags & 0x0C
            or function.__defaults__ is not None
            or function.__kwdefaults__ is not None
        ):
            raise RuntimeError("Package0 public raw ABI differs")
    if tuple(RESULT_FIELDS_IN_EXACT_ORDER_V1) != tuple(
        VerifiedPhase2BActualUnsealed960Package0MechanicsV1.__dataclass_fields__
    ):
        raise RuntimeError("Package0 private result fields differ")
    if tuple(
        (name, field.type)
        for name, field in VerifiedPhase2BActualUnsealed960Package0MechanicsV1.__dataclass_fields__.items()
    ) != RESULT_FIELD_TYPES_IN_EXACT_ORDER_V1:
        raise RuntimeError("Package0 private result field types differ")
    catalog_abi = _catalog.FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1
    if (
        tuple(
            tuple(row)
            for row in catalog_abi["positional_parameters_in_exact_order"]
        )
        != POSITIONAL_PARAMETERS_IN_EXACT_ORDER_V1
        or tuple(
            catalog_abi[
                "mandatory_keyword_only_byte_parameters_in_exact_order"
            ]
        )
        != KEYWORD_ONLY_PARAMETERS_IN_EXACT_ORDER_V1
        or tuple(
            row[0]
            for row in catalog_abi["verified_result_contract"][
                "fields_in_exact_order"
            ]
        )
        != RESULT_FIELDS_IN_EXACT_ORDER_V1
        or tuple(
            catalog_abi["verified_result_contract"]
            ["mechanics_boolean_fields_that_may_be_true_in_exact_order"]
        )
        != MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1
        or tuple(
            catalog_abi["verified_result_contract"]
            ["nonclaim_boolean_fields_required_exact_false_in_exact_order"]
        )
        != NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1
        or catalog_abi["defaults_allowed"] is not False
        or catalog_abi["varargs_or_kwargs_allowed"] is not False
        or catalog_abi[
            "decoded_mapping_path_key_callback_socket_clock_nonce_inputs_allowed"
        ]
        is not False
        or catalog_abi["implementation_status"] != "NOT_IMPLEMENTED"
    ):
        raise RuntimeError("Package0 mechanics and frozen catalog ABI differ")
    require_code = (
        require_verified_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1.__code__
    )
    if (
        require_code.co_posonlyargcount != 0
        or require_code.co_argcount != 1
        or require_code.co_kwonlyargcount != 0
        or require_code.co_flags & 0x0C
    ):
        raise RuntimeError("Package0 private require ABI differs")
    if (
        sum(count for _, count in VERIFIED_SIGNATURE_BREAKDOWN_IN_EXACT_ORDER_V1)
        != TOTAL_VERIFIED_SIGNATURE_COUNT_V1
        or sum(
            count
            for name, count in VERIFIED_SIGNATURE_BREAKDOWN_IN_EXACT_ORDER_V1
            if name.startswith("package0_")
        )
        != PACKAGE0_OWNED_SIGNATURE_COUNT_V1
    ):
        raise RuntimeError("Package0 signature accounting differs")
    if (
        len(set(STAGE_SUBJECT_FORBIDDEN_FORWARD_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1))
        != len(STAGE_SUBJECT_FORBIDDEN_FORWARD_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1)
        or len(set(STAGE_SUBJECT_FORBIDDEN_FORWARD_ARTIFACT_KINDS_IN_EXACT_ORDER_V1))
        != len(STAGE_SUBJECT_FORBIDDEN_FORWARD_ARTIFACT_KINDS_IN_EXACT_ORDER_V1)
        or not all(
            type(item) is str and item
            for item in STAGE_SUBJECT_FORBIDDEN_FORWARD_ARTIFACT_KINDS_IN_EXACT_ORDER_V1
        )
    ):
        raise RuntimeError("Package0 forbidden forward-reference vocabulary differs")
    if (
        len(TERMINAL_SUBJECT_ARTIFACT_KINDS_IN_EXACT_ORDER_V1) != 2
        or len(TERMINAL_SUBJECT_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1) != 2
        or len(set(TERMINAL_SUBJECT_ARTIFACT_KINDS_IN_EXACT_ORDER_V1)) != 2
        or len(set(TERMINAL_SUBJECT_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1)) != 2
    ):
        raise RuntimeError("Package0 terminal subject vocabulary differs")


_assert_public_abi_v1()
del _assert_public_abi_v1
del _build_private_boundary_v1


__all__ = [
    "CANONICALIZATION_V1",
    "CLAIM_LEVEL_V1",
    "CONTRACT_BODY_V1",
    "CONTRACT_ID_V1",
    "CONTRACT_PAYLOAD_SHA256_V1",
    "CONTRACT_PAYLOAD_V1",
    "CONTRACT_SCHEMA_VERSION_V1",
    "KEYWORD_ONLY_PARAMETERS_IN_EXACT_ORDER_V1",
    "GENERIC_CONTENT_IDENTIFIER_FULLMATCH_ASCII_REGEX_V1",
    "MAXIMUM_EXACT_BYTES_PARAMETER_OR_TUPLE_ITEM_BYTES_V1",
    "MAXIMUM_JSON_SAFE_INTEGER_V1",
    "MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1",
    "NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1",
    "PACKAGE0_OWNED_SIGNATURE_COUNT_V1",
    "POSITIONAL_PARAMETERS_IN_EXACT_ORDER_V1",
    "REAL_VERIFICATION_STATUS_V1",
    "RESULT_FIELDS_IN_EXACT_ORDER_V1",
    "RESULT_FIELD_TYPES_IN_EXACT_ORDER_V1",
    "SCHEMA_VERSION_V1",
    "STATUS_V1",
    "STAGE_SUBJECT_FORBIDDEN_FORWARD_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1",
    "STAGE_SUBJECT_FORBIDDEN_FORWARD_ARTIFACT_KINDS_IN_EXACT_ORDER_V1",
    "TERMINAL_SUBJECT_ARTIFACT_KINDS_IN_EXACT_ORDER_V1",
    "TERMINAL_SUBJECT_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1",
    "TOTAL_VERIFIED_SIGNATURE_COUNT_V1",
    "VERIFIED_SIGNATURE_BREAKDOWN_IN_EXACT_ORDER_V1",
    "Package0MechanicsDispositionV1",
    "Package0MechanicsReasonV1",
    "Phase2BActualUnsealed960Package0MechanicsErrorV1",
    "VerifiedPhase2BActualUnsealed960Package0MechanicsV1",
    "canonical_phase2b_actual_unsealed_960_package0_mechanics_bytes_v1",
    "require_verified_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1",
    "verify_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1",
    "verify_real_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1",
]
