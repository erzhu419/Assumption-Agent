"""Downstream presented mechanics for the frozen Phase-2B R01 dispatch.

The authoritative-evidence catalog v1 freezes the dispatch identifier
``VERIFY_R01_REAL_CURRENT_CUT_ACTIVE_NONREVOKED_V1`` and an exact 24-item
native-source order, but it deliberately keeps that dispatch unavailable and
its verifier-source digest null.  This module does not edit or backfill that
catalog.  It defines a separate, versioned, local envelope/leaf contract that
can verify a fully presented synthetic byte graph in the catalog order.

Every composite source is accompanied by all of its raw terminal leaves.
Identifiers, digests, byte counts, or a composite manifest without those raw
leaves are rejected.  Success establishes presented mechanics only.  It does
not establish currentness, authority, owner non-revocation, R01, readiness,
execution, an actual-960 run, or C1.  The same-ABI real entry remains exactly
``NOT_IMPLEMENTED``.
"""

from __future__ import annotations

import builtins as _builtins
from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from types import FunctionType, ModuleType
from typing import Final, NoReturn

from . import (
    phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1
    as _catalog,
)
from . import rfc8785_v1 as _rfc8785


SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-verify-r01-real-current-cut-"
    "active-nonrevoked-presented-mechanics/1"
)
STATUS_V1: Final = (
    "VERIFIED_PRESENTED_R01_24_SOURCE_FULL_RAW_LEAF_MECHANICS_NOT_R01"
)
CLAIM_LEVEL_V1: Final = (
    "LOCAL_OFFLINE_PRODUCER_FREE_PRESENTED_MECHANICS_ONLY_NOT_CURRENTNESS_"
    "NOT_AUTHORITY_NOT_OWNER_NONREVOCATION_NOT_R01_NOT_READINESS_NOT_"
    "EXECUTION_NOT_ACTUAL960_NOT_C1"
)
CANONICALIZATION_V1: Final = "RFC8785_EXACT_BYTES_NO_TRAILING_LF"
REAL_VERIFICATION_STATUS_V1: Final = "NOT_IMPLEMENTED"

CATALOG_R01_GATE_ORDINAL_V1: Final = 1
CATALOG_R01_GATE_ID_V1: Final = "R01_AUTHORIZATION_VERIFIED"
CATALOG_R01_DISPATCH_ID_V1: Final = (
    "VERIFY_R01_REAL_CURRENT_CUT_ACTIVE_NONREVOKED_V1"
)
CATALOG_R01_AVAILABILITY_V1: Final = "SCHEMA_FROZEN_NOT_VERIFIABLE"
INSPECTED_CATALOG_SOURCE_SHA256_V1: Final = (
    "ead2e2e5a7b03a804148fbd329e67561c1367375b2c05bf4ab2cd691f20487c6"
)
INSPECTED_CATALOG_CONFIG_SHA256_V1: Final = (
    "62f46cfba8a86f7ad969a7ea584c1828f0b34bb1d95571b1cc64259ac01a588a"
)
INSPECTED_RFC8785_SOURCE_SHA256_V1: Final = (
    "16dcc6524a1c4a507edb9f85455c72b00fbc6c0c1b67471f03b67b2f25d32654"
)

R01_NATIVE_SOURCE_KINDS_IN_EXACT_ORDER_V1: Final = (
    "SIGNED_OWNER_AUTHORIZATION",
    "STAGE_B_TRUST_LOCK",
    "AUTHORITY_PROFILE",
    "SIGNER_REGISTRY",
    "OWNER_STATUS_LEDGER_BUNDLE",
    "CURRENT_CUT_RESPONSE",
    "CURRENT_CUT_STATEMENT",
    "CURRENT_CUT_LOGICAL_TIME_CHAIN",
    "EXTERNAL_WITNESS_POLICY",
    "SIGNED_ANTI_FORK_ANCHOR",
    "EXTERNAL_ANCHOR_RECEIPT",
    "LIVE_RUNTIME_CLOSURE_REPORT_0",
    "LIVE_RUNTIME_CLOSURE_REPORT_1",
    "LIVE_RUNTIME_CLOSURE_REPORT_2",
    "LIVE_RUNTIME_CLOSURE_REPORT_3",
    "LIVE_RUNTIME_CLOSURE_REPORT_4",
    "LIVE_RUNTIME_CLOSURE_BUNDLE",
    "SERVICE_KEY_CUSTODY_ATTESTATION_0",
    "SERVICE_KEY_CUSTODY_ATTESTATION_1",
    "SERVICE_KEY_CUSTODY_ATTESTATION_2",
    "SERVICE_KEY_CUSTODY_ATTESTATION_3",
    "SERVICE_KEY_CUSTODY_ATTESTATION_4",
    "CAS_COMPLETE_SESSION",
    "DUAL_SIGNED_R01_RECEIPT",
)

R01_NATIVE_SOURCE_PREDECESSOR_ORDINALS_IN_EXACT_ORDER_V1: Final = (
    (),
    (0,),
    (1,),
    (2,),
    (0, 1, 2, 3),
    (4,),
    (5,),
    (5, 6),
    (3,),
    (6, 7, 8),
    (9,),
    (2, 3),
    (2, 3),
    (2, 3),
    (2, 3),
    (2, 3),
    (11, 12, 13, 14, 15),
    (3, 11),
    (3, 12),
    (3, 13),
    (3, 14),
    (3, 15),
    (4, 7, 16, 17, 18, 19, 20, 21),
    tuple(range(23)),
)

R01_PRESENTED_STATE_PROFILES_IN_EXACT_ORDER_V1: Final = (
    "PRESENTED_OWNER_AUTHORIZATION_SIGNATURE_MECHANICS",
    "PRESENTED_STAGE_B_TRUST_LOCK_MECHANICS",
    "PRESENTED_AUTHORITY_PROFILE_BINDING_MECHANICS",
    "PRESENTED_SIGNER_REGISTRY_BINDING_MECHANICS",
    "PRESENTED_OWNER_STATUS_ACTIVE_ZERO_REVOCATION_NONTERMINAL_MECHANICS",
    "PRESENTED_CURRENT_CUT_RESPONSE_ACTIVE_ZERO_REVOCATION_NONTERMINAL_MECHANICS",
    "PRESENTED_CURRENT_CUT_STATEMENT_BINDING_MECHANICS",
    "PRESENTED_CURRENT_CUT_LOGICAL_TIME_CHAIN_MECHANICS",
    "PRESENTED_EXTERNAL_WITNESS_POLICY_MECHANICS",
    "PRESENTED_SIGNED_ANTI_FORK_ANCHOR_MECHANICS",
    "PRESENTED_EXTERNAL_ANCHOR_RECEIPT_MECHANICS",
    "PRESENTED_LIVE_RUNTIME_CLOSURE_REPORT_0_MECHANICS",
    "PRESENTED_LIVE_RUNTIME_CLOSURE_REPORT_1_MECHANICS",
    "PRESENTED_LIVE_RUNTIME_CLOSURE_REPORT_2_MECHANICS",
    "PRESENTED_LIVE_RUNTIME_CLOSURE_REPORT_3_MECHANICS",
    "PRESENTED_LIVE_RUNTIME_CLOSURE_REPORT_4_MECHANICS",
    "PRESENTED_FIVE_REPORT_RUNTIME_BUNDLE_MECHANICS",
    "PRESENTED_SERVICE_KEY_CUSTODY_ATTESTATION_0_MECHANICS",
    "PRESENTED_SERVICE_KEY_CUSTODY_ATTESTATION_1_MECHANICS",
    "PRESENTED_SERVICE_KEY_CUSTODY_ATTESTATION_2_MECHANICS",
    "PRESENTED_SERVICE_KEY_CUSTODY_ATTESTATION_3_MECHANICS",
    "PRESENTED_SERVICE_KEY_CUSTODY_ATTESTATION_4_MECHANICS",
    "PRESENTED_MAIN_CAS_29_OBJECT_COMPLETE_SESSION_MECHANICS",
    "PRESENTED_DUAL_SIGNED_R01_RECEIPT_AND_RAW_BODY_MECHANICS_NOT_R01",
)

OWNER_STATUS_LEDGER_LEAF_KINDS_IN_EXACT_ORDER_V1: Final = (
    "OWNER_STATUS_GENESIS_HEAD",
    "OWNER_STATUS_ACTIVE_DECISION",
    "OWNER_STATUS_ACTIVE_SIGNED_EVENT",
    "OWNER_STATUS_ACTIVE_CHECKPOINT",
    "OWNER_STATUS_ACTIVE_HEAD",
    "OWNER_STATUS_ACTIVE_PROJECTION",
)
CURRENT_CUT_LOGICAL_TIME_LEAF_KINDS_IN_EXACT_ORDER_V1: Final = (
    "CURRENT_CUT_LOGICAL_TIME_GENESIS_HEAD",
    "CURRENT_CUT_LOGICAL_TIME_REQUEST",
    "CURRENT_CUT_LOGICAL_TIME_SIGNED_TOKEN",
    "CURRENT_CUT_LOGICAL_TIME_CHECKPOINT",
    "CURRENT_CUT_LOGICAL_TIME_CURRENT_HEAD",
)
RUNTIME_CLOSURE_BUNDLE_LEAF_KINDS_IN_EXACT_ORDER_V1: Final = tuple(
    f"LIVE_RUNTIME_CLOSURE_REPORT_{index}" for index in range(5)
)
MAIN_CAS_OBJECT_COUNT_V1: Final = 29
MAIN_CAS_LEAF_KINDS_IN_EXACT_ORDER_V1: Final = (
    ("CAS_RETENTION_PLAN",)
    + tuple(f"CAS_REQUEST_{index:03d}" for index in range(MAIN_CAS_OBJECT_COUNT_V1))
    + tuple(f"CAS_OBJECT_{index:03d}" for index in range(MAIN_CAS_OBJECT_COUNT_V1))
    + tuple(f"CAS_SIGNED_RECEIPT_{index:03d}" for index in range(MAIN_CAS_OBJECT_COUNT_V1))
    + tuple(f"CAS_CHECKPOINT_{index:03d}" for index in range(MAIN_CAS_OBJECT_COUNT_V1))
    + tuple(f"CAS_HEAD_{index:03d}" for index in range(MAIN_CAS_OBJECT_COUNT_V1 + 1))
    + tuple(f"CAS_RESPONSE_{index:03d}" for index in range(MAIN_CAS_OBJECT_COUNT_V1))
)
DUAL_SIGNED_R01_RECEIPT_LEAF_KINDS_IN_EXACT_ORDER_V1: Final = (
    "R01_RECEIPT_BODY",
)
COMPOSITE_LEAF_KINDS_BY_SOURCE_ORDINAL_V1: Final = {
    4: OWNER_STATUS_LEDGER_LEAF_KINDS_IN_EXACT_ORDER_V1,
    7: CURRENT_CUT_LOGICAL_TIME_LEAF_KINDS_IN_EXACT_ORDER_V1,
    16: RUNTIME_CLOSURE_BUNDLE_LEAF_KINDS_IN_EXACT_ORDER_V1,
    22: MAIN_CAS_LEAF_KINDS_IN_EXACT_ORDER_V1,
    23: DUAL_SIGNED_R01_RECEIPT_LEAF_KINDS_IN_EXACT_ORDER_V1,
}
COMPOSITE_SOURCE_ORDINALS_IN_EXACT_ORDER_V1: Final = tuple(
    COMPOSITE_LEAF_KINDS_BY_SOURCE_ORDINAL_V1
)
TOTAL_REQUIRED_RAW_COMPOSITE_LEAF_COUNT_V1: Final = sum(
    len(value) for value in COMPOSITE_LEAF_KINDS_BY_SOURCE_ORDINAL_V1.values()
)

NATIVE_SOURCE_ENVELOPE_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-r01-native-source-envelope/1"
)
NATIVE_SOURCE_BODY_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-r01-native-source-body/1"
)
RAW_COMPOSITE_LEAF_ENVELOPE_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-r01-raw-composite-leaf/1"
)
RAW_COMPOSITE_LEAF_BODY_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-r01-raw-composite-leaf-body/1"
)
COMPOSITE_MANIFEST_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-r01-composite-manifest/1"
)
NATIVE_SOURCE_STATUS_V1: Final = "PRESENTED_MECHANICS_ONLY"
NATIVE_SOURCE_CLAIM_LEVEL_V1: Final = CLAIM_LEVEL_V1
RAW_LEAF_STATUS_V1: Final = "PRESENTED_RAW_TERMINAL_LEAF_MECHANICS_ONLY"
RAW_LEAF_CLAIM_LEVEL_V1: Final = CLAIM_LEVEL_V1
NATIVE_SOURCE_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_r01_native_source_v1_"
)
RAW_COMPOSITE_LEAF_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_r01_raw_composite_leaf_v1_"
)
NATIVE_SOURCE_ID_DOMAIN_PREFIX_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/R01/NATIVE_SOURCE/"
)
RAW_COMPOSITE_LEAF_ID_DOMAIN_PREFIX_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/R01/RAW_COMPOSITE_LEAF/"
)
COMPOSITE_LEAF_ROOT_DOMAIN_PREFIX_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/R01/COMPOSITE_LEAF_ROOT/"
)
SOURCE_GRAPH_ROOT_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/R01/SOURCE_GRAPH_ROOT/V1\0"
)

NATIVE_SOURCE_ENVELOPE_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "schema_version",
    "body",
    "native_source_id",
)
NATIVE_SOURCE_BODY_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "schema_version",
    "status",
    "claim_level",
    "catalog_id",
    "catalog_payload_sha256",
    "gate_ordinal",
    "gate_id",
    "verifier_dispatch_id",
    "native_source_ordinal",
    "native_source_kind",
    "predecessor_ordinals_in_exact_order",
    "predecessor_source_ids_in_exact_order",
    "predecessor_source_payload_sha256s_in_exact_order",
    "presented_state_profile",
    "composite_leaf_manifest_or_none",
    "nonclaims",
)
RAW_COMPOSITE_LEAF_ENVELOPE_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "schema_version",
    "body",
    "raw_leaf_id",
)
RAW_COMPOSITE_LEAF_BODY_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "schema_version",
    "status",
    "claim_level",
    "parent_source_ordinal",
    "parent_source_kind",
    "leaf_ordinal",
    "leaf_kind",
    "payload_label",
    "nonclaims",
)
COMPOSITE_MANIFEST_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "schema_version",
    "profile_id",
    "leaf_kinds_in_exact_order",
    "leaf_ids_in_exact_order",
    "leaf_payload_sha256s_in_exact_order",
    "leaf_payload_byte_counts_in_exact_order",
    "leaf_root_sha256",
)

NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "external_currentness_verified",
    "owner_non_revocation_verified",
    "authority_verified",
    "gate_admission_verified",
    "r01_satisfied",
    "readiness_verified",
    "execution_authorized",
    "actual_960_verified",
    "c1_satisfied",
)
NONCLAIM_ORACLE_V1: Final = {
    name: False for name in NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1
}
MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "exact_raw_transport_verified",
    "catalog_r01_descriptor_verified",
    "rfc8785_envelopes_verified",
    "source_content_ids_verified",
    "native_source_order_verified",
    "source_predecessor_graph_verified",
    "raw_payload_sha256s_and_byte_counts_verified",
    "composite_manifests_verified",
    "full_raw_composite_leaf_expansion_verified",
    "presented_active_zero_revocation_profile_literal_mechanics_verified",
    "permanent_nonclaims_verified",
    "private_lineage_verified",
)

POSITIONAL_PARAMETERS_IN_EXACT_ORDER_V1: Final = (
    ("catalog_payload", "EXACT_BUILTIN_BYTES"),
    (
        "raw_native_source_payloads_in_exact_order",
        "EXACT_BUILTIN_TUPLE_OF_24_EXACT_BUILTIN_BYTES",
    ),
    (
        "raw_composite_leaf_payload_tuples_in_exact_native_source_order",
        "EXACT_BUILTIN_TUPLE_OF_24_EXACT_BUILTIN_TUPLES_OF_EXACT_BUILTIN_BYTES",
    ),
)
KEYWORD_ONLY_PARAMETERS_IN_EXACT_ORDER_V1: Final = (
    "out_of_band_expected_catalog_id_bytes",
    "out_of_band_expected_catalog_payload_sha256_digest",
)
MAXIMUM_EXACT_BYTES_PARAMETER_OR_TUPLE_ITEM_BYTES_V1: Final = 16_777_216
MAXIMUM_JSON_SAFE_INTEGER_V1: Final = (1 << 53) - 1

FAIL_TYPE_V1: Final = "FAIL_R01_PRESENTED_MECHANICS_V1_TYPE"
FAIL_CATALOG_V1: Final = "FAIL_R01_PRESENTED_MECHANICS_V1_CATALOG"
FAIL_CANONICAL_V1: Final = "FAIL_R01_PRESENTED_MECHANICS_V1_CANONICAL"
FAIL_SCHEMA_V1: Final = "FAIL_R01_PRESENTED_MECHANICS_V1_SCHEMA"
FAIL_CONTENT_ID_V1: Final = "FAIL_R01_PRESENTED_MECHANICS_V1_CONTENT_ID"
FAIL_BINDING_V1: Final = "FAIL_R01_PRESENTED_MECHANICS_V1_BINDING"
FAIL_COMPOSITE_V1: Final = "FAIL_R01_PRESENTED_MECHANICS_V1_COMPOSITE"
FAIL_CLAIM_BOUNDARY_V1: Final = (
    "FAIL_R01_PRESENTED_MECHANICS_V1_CLAIM_BOUNDARY"
)
FAIL_RESULT_V1: Final = "FAIL_R01_PRESENTED_MECHANICS_V1_RESULT"
FAIL_NOT_IMPLEMENTED_V1: Final = (
    "FAIL_R01_REAL_CURRENT_CUT_ACTIVE_NONREVOKED_V1_NOT_IMPLEMENTED"
)


class R01PresentedMechanicsDispositionV1(str, Enum):
    VERIFIED_PRESENTED_R01_MECHANICS_NOT_R01 = (
        "VERIFIED_PRESENTED_R01_MECHANICS_NOT_R01"
    )


class R01PresentedMechanicsReasonV1(str, Enum):
    EXACT_24_SOURCE_AND_FULL_RAW_LEAF_GRAPH_VERIFIED = (
        "EXACT_24_SOURCE_AND_FULL_RAW_LEAF_GRAPH_VERIFIED"
    )


class Phase2BActualUnsealed960R01PresentedMechanicsErrorV1(ValueError):
    """Stable fail-closed error for the downstream presented verifier."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True, init=False)
class VerifiedPhase2BActualUnsealed960R01PresentedMechanicsV1:
    schema_version: str
    status: str
    claim_level: str
    disposition: R01PresentedMechanicsDispositionV1
    reason: R01PresentedMechanicsReasonV1
    contract_id: str
    catalog_id: str
    catalog_payload_sha256: str
    gate_ordinal: int
    gate_id: str
    verifier_dispatch_id: str
    native_source_count: int
    composite_source_count: int
    raw_composite_leaf_count: int
    source_graph_root_sha256: str
    native_source_ids_in_exact_order: tuple[str, ...]
    native_source_payload_sha256s_in_exact_order: tuple[str, ...]
    exact_raw_transport_verified: bool
    catalog_r01_descriptor_verified: bool
    rfc8785_envelopes_verified: bool
    source_content_ids_verified: bool
    native_source_order_verified: bool
    source_predecessor_graph_verified: bool
    raw_payload_sha256s_and_byte_counts_verified: bool
    composite_manifests_verified: bool
    full_raw_composite_leaf_expansion_verified: bool
    presented_active_zero_revocation_profile_literal_mechanics_verified: bool
    permanent_nonclaims_verified: bool
    private_lineage_verified: bool
    external_currentness_verified: bool
    owner_non_revocation_verified: bool
    authority_verified: bool
    gate_admission_verified: bool
    r01_satisfied: bool
    readiness_verified: bool
    execution_authorized: bool
    actual_960_verified: bool
    c1_satisfied: bool

    def __new__(
        cls, *args: object, **kwargs: object
    ) -> "VerifiedPhase2BActualUnsealed960R01PresentedMechanicsV1":
        raise TypeError("R01 presented-mechanics results are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("R01 presented-mechanics results are privately issued")

    def __copy__(self) -> NoReturn:
        raise TypeError("R01 presented-mechanics results cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("R01 presented-mechanics results cannot be deep-copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("R01 presented-mechanics results cannot be serialized")


RESULT_FIELDS_IN_EXACT_ORDER_V1: Final = tuple(
    VerifiedPhase2BActualUnsealed960R01PresentedMechanicsV1.__dataclass_fields__
)
RESULT_FIELD_TYPES_IN_EXACT_ORDER_V1: Final = tuple(
    (name, field.type)
    for name, field in (
        VerifiedPhase2BActualUnsealed960R01PresentedMechanicsV1
        .__dataclass_fields__
        .items()
    )
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
    """Load the pinned local RFC8785 bytes into a private function graph."""

    source_path = getattr(imported_module, "__file__", None)
    if type(source_path) is not str or not source_path:
        raise RuntimeError("R01 downstream RFC8785 source path is unavailable")
    try:
        with _builtins.open(source_path, "rb") as handle:
            source_bytes = handle.read()
    except OSError as exc:
        raise RuntimeError("R01 downstream RFC8785 source bytes are unavailable") from exc
    if hashlib.sha256(source_bytes).hexdigest() != INSPECTED_RFC8785_SOURCE_SHA256_V1:
        raise RuntimeError("R01 downstream RFC8785 source bytes differ")

    private_module = ModuleType(
        "hegel_machine._phase2b_actual_unsealed_960_r01_frozen_rfc8785_v1"
    )
    private_namespace = vars(private_module)
    private_namespace["__file__"] = source_path
    private_namespace["__package__"] = "hegel_machine"
    private_namespace["__builtins__"] = dict(vars(_builtins))
    code = _builtins.compile(source_bytes, source_path, "exec", dont_inherit=True)
    _builtins.exec(code, private_namespace)
    return _capture_dependency_function_graph_v1(private_module)


_RFC8785_FUNCTIONS_V1 = _load_exact_rfc8785_function_graph_v1(_rfc8785)
_CAPTURED_CANONICALIZE_V1: Final = _RFC8785_FUNCTIONS_V1[
    "canonicalize_rfc8785_v1"
]
_CAPTURED_DECODE_V1: Final = _RFC8785_FUNCTIONS_V1[
    "decode_and_require_canonical_rfc8785_v1"
]
del _RFC8785_FUNCTIONS_V1
del _load_exact_rfc8785_function_graph_v1
del _capture_dependency_function_graph_v1


def _assert_catalog_boundary_v1() -> None:
    payload = _catalog.CATALOG_PAYLOAD_V1
    if type(payload) is not bytes:
        raise RuntimeError("R01 downstream catalog payload type differs")
    if hashlib.sha256(payload).hexdigest() != INSPECTED_CATALOG_CONFIG_SHA256_V1:
        raise RuntimeError("R01 downstream catalog payload SHA differs")
    if _catalog.CATALOG_ID_V1 != (
        "phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1_"
        "a9f1b10bf7df69065f9847a69b935613ba1c368832c21ea5ac60e863961272cd"
    ):
        raise RuntimeError("R01 downstream catalog ID differs")
    decoded = _CAPTURED_DECODE_V1(payload)
    if type(decoded) is not dict or set(decoded) != {
        "schema_version", "body", "catalog_id"
    }:
        raise RuntimeError("R01 downstream catalog root differs")
    body = decoded["body"]
    if type(body) is not dict:
        raise RuntimeError("R01 downstream catalog body differs")
    sources = body.get("gate_source_schema_objects_in_exact_order")
    if type(sources) is not list or len(sources) != 16:
        raise RuntimeError("R01 downstream source descriptor set differs")
    source = sources[0]
    if (
        type(source) is not dict
        or source.get("gate_ordinal") != CATALOG_R01_GATE_ORDINAL_V1
        or source.get("gate_id") != CATALOG_R01_GATE_ID_V1
        or source.get("verifier_dispatch_id") != CATALOG_R01_DISPATCH_ID_V1
        or source.get("verification_availability")
        != CATALOG_R01_AVAILABILITY_V1
        or source.get("verifier_source_sha256_or_none") is not None
        or source.get("allowed_native_source_kind_orders_in_exact_order")
        != [list(R01_NATIVE_SOURCE_KINDS_IN_EXACT_ORDER_V1)]
        or source.get("allowed_native_source_cardinalities_in_exact_order")
        != [24]
    ):
        raise RuntimeError("R01 downstream catalog descriptor differs")
    public_spec = _catalog.GATE_SOURCE_SPECS_IN_EXACT_ORDER_V1[0]
    if (
        public_spec.ordinal != CATALOG_R01_GATE_ORDINAL_V1
        or public_spec.gate_id != CATALOG_R01_GATE_ID_V1
        or public_spec.verifier_dispatch_id != CATALOG_R01_DISPATCH_ID_V1
        or public_spec.verification_availability
        != CATALOG_R01_AVAILABILITY_V1
        or public_spec.verifier_source_sha256_or_none is not None
        or public_spec.allowed_native_source_kind_orders_in_exact_order
        != (R01_NATIVE_SOURCE_KINDS_IN_EXACT_ORDER_V1,)
    ):
        raise RuntimeError("R01 downstream mutable catalog surface differs")


_assert_catalog_boundary_v1()
del _assert_catalog_boundary_v1


def canonical_phase2b_actual_unsealed_960_r01_presented_mechanics_bytes_v1(
    value: object,
    _canonicalize: FunctionType = _CAPTURED_CANONICALIZE_V1,
) -> bytes:
    """Return the downstream contract's exact RFC8785 encoding."""

    return _canonicalize(value)


CONTRACT_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-verify-r01-real-current-cut-"
    "active-nonrevoked-contract/1"
)
CONTRACT_ID_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/VERIFY_R01/PRESENTED_CONTRACT/ID/V1\0"
)
CONTRACT_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_verify_r01_presented_contract_v1_"
)
CONTRACT_BODY_V1: Final = {
    "schema_version": CONTRACT_SCHEMA_VERSION_V1,
    "status": "DOWNSTREAM_PRESENTED_MECHANICS_IMPLEMENTED_REAL_NOT_IMPLEMENTED",
    "claim_level": CLAIM_LEVEL_V1,
    "canonicalization": CANONICALIZATION_V1,
    "catalog_v1_modified_or_reissued": False,
    "catalog_dispatch_registration_performed": False,
    "catalog_r01_availability_remains": CATALOG_R01_AVAILABILITY_V1,
    "catalog_r01_verifier_source_sha256_remains": None,
    "catalog_r01_gate_success_allowed": False,
    "inspected_catalog_id": _catalog.CATALOG_ID_V1,
    "inspected_catalog_payload_sha256": (
        "sha256:" + INSPECTED_CATALOG_CONFIG_SHA256_V1
    ),
    "inspected_catalog_source_sha256": INSPECTED_CATALOG_SOURCE_SHA256_V1,
    "inspected_catalog_config_sha256": INSPECTED_CATALOG_CONFIG_SHA256_V1,
    "inspected_rfc8785_source_sha256": INSPECTED_RFC8785_SOURCE_SHA256_V1,
    "rfc8785_dependency_capture_profile": (
        "DEFINITION_TIME_READ_ONLY_EXACT_LOCAL_SOURCE_BYTES_SHA256_"
        "THEN_PRIVATE_REEXECUTION_AND_FUNCTION_GRAPH_CAPTURE"
    ),
    "rfc8785_dependency_source_path_is_caller_input": False,
    "rfc8785_dependency_source_check_claims_currentness_or_authority": False,
    "catalog_gate_ordinal": CATALOG_R01_GATE_ORDINAL_V1,
    "catalog_gate_id": CATALOG_R01_GATE_ID_V1,
    "catalog_dispatch_id": CATALOG_R01_DISPATCH_ID_V1,
    "native_source_kinds_in_exact_order": list(
        R01_NATIVE_SOURCE_KINDS_IN_EXACT_ORDER_V1
    ),
    "native_source_predecessor_ordinals_in_exact_order": [
        list(value)
        for value in R01_NATIVE_SOURCE_PREDECESSOR_ORDINALS_IN_EXACT_ORDER_V1
    ],
    "presented_state_profiles_in_exact_order": list(
        R01_PRESENTED_STATE_PROFILES_IN_EXACT_ORDER_V1
    ),
    "composite_source_ordinals_in_exact_order": list(
        COMPOSITE_SOURCE_ORDINALS_IN_EXACT_ORDER_V1
    ),
    "composite_leaf_kinds_by_source_ordinal": {
        str(ordinal): list(COMPOSITE_LEAF_KINDS_BY_SOURCE_ORDINAL_V1[ordinal])
        for ordinal in COMPOSITE_SOURCE_ORDINALS_IN_EXACT_ORDER_V1
    },
    "main_cas_object_count": MAIN_CAS_OBJECT_COUNT_V1,
    "main_cas_full_raw_leaf_count": len(MAIN_CAS_LEAF_KINDS_IN_EXACT_ORDER_V1),
    "total_required_raw_composite_leaf_count": (
        TOTAL_REQUIRED_RAW_COMPOSITE_LEAF_COUNT_V1
    ),
    "native_source_envelope_contract": {
        "canonical_wire": CANONICALIZATION_V1,
        "envelope_fields_in_exact_order": list(
            NATIVE_SOURCE_ENVELOPE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "body_fields_in_exact_order": list(
            NATIVE_SOURCE_BODY_FIELDS_IN_EXACT_ORDER_V1
        ),
        "envelope_schema_version": NATIVE_SOURCE_ENVELOPE_SCHEMA_VERSION_V1,
        "body_schema_version": NATIVE_SOURCE_BODY_SCHEMA_VERSION_V1,
        "status": NATIVE_SOURCE_STATUS_V1,
        "claim_level": NATIVE_SOURCE_CLAIM_LEVEL_V1,
        "content_id_prefix": NATIVE_SOURCE_ID_PREFIX_V1,
        "content_id_domain_prefix_bytes_hex": (
            NATIVE_SOURCE_ID_DOMAIN_PREFIX_V1.hex()
        ),
        "content_id_domain_suffix_bytes_hex": b"/ID/V1\0".hex(),
        "content_id_domain_formula": (
            "ASCII(HEGEL/PHASE2B/ACTUAL960/R01/NATIVE_SOURCE/) || "
            "ASCII_DECIMAL(native_source_ordinal) || ASCII(/ID/V1) || NUL"
        ),
        "content_id_digest_preimage_formula": (
            "domain_bytes || RFC8785_EXACT_BYTES(body)"
        ),
        "content_id_formula": "prefix || LOWER_HEX(SHA256(preimage))",
        "payload_sha256_formula": (
            "ASCII(sha256:) || LOWER_HEX(SHA256(raw_native_source_bytes))"
        ),
    },
    "raw_composite_leaf_envelope_contract": {
        "canonical_wire": CANONICALIZATION_V1,
        "envelope_fields_in_exact_order": list(
            RAW_COMPOSITE_LEAF_ENVELOPE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "body_fields_in_exact_order": list(
            RAW_COMPOSITE_LEAF_BODY_FIELDS_IN_EXACT_ORDER_V1
        ),
        "envelope_schema_version": RAW_COMPOSITE_LEAF_ENVELOPE_SCHEMA_VERSION_V1,
        "body_schema_version": RAW_COMPOSITE_LEAF_BODY_SCHEMA_VERSION_V1,
        "status": RAW_LEAF_STATUS_V1,
        "claim_level": RAW_LEAF_CLAIM_LEVEL_V1,
        "payload_label_formula": "EXACT_LEAF_KIND_LITERAL",
        "content_id_prefix": RAW_COMPOSITE_LEAF_ID_PREFIX_V1,
        "content_id_domain_prefix_bytes_hex": (
            RAW_COMPOSITE_LEAF_ID_DOMAIN_PREFIX_V1.hex()
        ),
        "content_id_domain_ordinal_separator_bytes_hex": b"/".hex(),
        "content_id_domain_suffix_bytes_hex": b"/ID/V1\0".hex(),
        "content_id_domain_formula": (
            "ASCII(HEGEL/PHASE2B/ACTUAL960/R01/RAW_COMPOSITE_LEAF/) || "
            "ASCII_DECIMAL(parent_source_ordinal) || ASCII(/) || "
            "ASCII_DECIMAL(leaf_ordinal) || ASCII(/ID/V1) || NUL"
        ),
        "content_id_digest_preimage_formula": (
            "domain_bytes || RFC8785_EXACT_BYTES(body)"
        ),
        "content_id_formula": "prefix || LOWER_HEX(SHA256(preimage))",
        "payload_sha256_formula": (
            "ASCII(sha256:) || LOWER_HEX(SHA256(raw_leaf_bytes))"
        ),
        "payload_byte_count_formula": "LEN(raw_leaf_bytes)",
    },
    "composite_manifest_contract": {
        "schema_version": COMPOSITE_MANIFEST_SCHEMA_VERSION_V1,
        "fields_in_exact_order": list(
            COMPOSITE_MANIFEST_FIELDS_IN_EXACT_ORDER_V1
        ),
        "profile_id_formula": (
            "R01_SOURCE_ || ZERO_PAD_2(source_ordinal) || _FULL_RAW_LEAVES_V1"
        ),
        "leaf_root_domain_prefix_bytes_hex": (
            COMPOSITE_LEAF_ROOT_DOMAIN_PREFIX_V1.hex()
        ),
        "leaf_root_domain_suffix_bytes_hex": b"/V1\0".hex(),
        "leaf_root_domain_formula": (
            "ASCII(HEGEL/PHASE2B/ACTUAL960/R01/COMPOSITE_LEAF_ROOT/) || "
            "ASCII_DECIMAL(source_ordinal) || ASCII(/V1) || NUL"
        ),
        "leaf_root_preimage_fields_in_exact_order": [
            "leaf_kinds_in_exact_order",
            "leaf_ids_in_exact_order",
            "leaf_payload_sha256s_in_exact_order",
            "leaf_payload_byte_counts_in_exact_order",
        ],
        "leaf_root_formula": (
            "ASCII(sha256:) || LOWER_HEX(SHA256(domain_bytes || "
            "RFC8785_EXACT_BYTES(preimage_object)))"
        ),
    },
    "source_graph_root_contract": {
        "domain_bytes_hex": SOURCE_GRAPH_ROOT_DOMAIN_V1.hex(),
        "preimage_fields_in_exact_order": [
            "native_source_kinds_in_exact_order",
            "native_source_ids_in_exact_order",
            "native_source_payload_sha256s_in_exact_order",
            "composite_leaf_roots_in_exact_source_order",
        ],
        "formula": (
            "ASCII(sha256:) || LOWER_HEX(SHA256(domain_bytes || "
            "RFC8785_EXACT_BYTES(preimage_object)))"
        ),
    },
    "raw_leaf_matrix_contract": {
        "outer_rows": 24,
        "row_order": "EXACT_NATIVE_SOURCE_ORDER",
        "atomic_source_row": "EXACT_EMPTY_BUILTIN_TUPLE",
        "composite_source_row": "EXACT_FULL_RAW_BYTES_IN_FROZEN_LEAF_ORDER",
        "non_runtime_composite_leaf_wire": (
            "RAW_COMPOSITE_LEAF_ENVELOPE_CANONICAL_BYTES"
        ),
        "runtime_bundle_leaf_wire": (
            "BYTE_EQUAL_NATIVE_SOURCE_11_THROUGH_15_IN_ORDER"
        ),
        "runtime_bundle_manifest_leaf_ids_formula": (
            "NATIVE_SOURCE_IDS_11_THROUGH_15_IN_ORDER"
        ),
        "runtime_bundle_manifest_leaf_payload_sha256s_formula": (
            "NATIVE_SOURCE_PAYLOAD_SHA256S_11_THROUGH_15_IN_ORDER"
        ),
        "runtime_bundle_manifest_leaf_payload_byte_counts_formula": (
            "LEN(RAW_NATIVE_SOURCE_BYTES_11_THROUGH_15)_IN_ORDER"
        ),
        "total_required_raw_leaves": TOTAL_REQUIRED_RAW_COMPOSITE_LEAF_COUNT_V1,
    },
    "opaque_id_or_sha_only_composite_accepted": False,
    "runtime_bundle_leaves_must_equal_native_sources_11_through_15": True,
    "dual_receipt_requires_raw_receipt_body_leaf": True,
    "positional_parameters_in_exact_order": [
        [name, wire_type] for name, wire_type in POSITIONAL_PARAMETERS_IN_EXACT_ORDER_V1
    ],
    "mandatory_keyword_only_parameters_in_exact_order": list(
        KEYWORD_ONLY_PARAMETERS_IN_EXACT_ORDER_V1
    ),
    "defaults_allowed": False,
    "varargs_or_kwargs_allowed": False,
    "decoded_mapping_path_key_callback_socket_clock_nonce_inputs_allowed": False,
    "maximum_exact_bytes_parameter_or_tuple_item_bytes": (
        MAXIMUM_EXACT_BYTES_PARAMETER_OR_TUPLE_ITEM_BYTES_V1
    ),
    "mechanics_boolean_fields_in_exact_order": list(
        MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1
    ),
    "nonclaim_boolean_fields_required_false_in_exact_order": list(
        NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1
    ),
    "private_result_fields_in_exact_order": list(RESULT_FIELDS_IN_EXACT_ORDER_V1),
    "private_result_contract": {
        "type_name": (
            "VerifiedPhase2BActualUnsealed960R01PresentedMechanicsV1"
        ),
        "fields_and_types_in_exact_order": [
            [name, field_type]
            for name, field_type in RESULT_FIELD_TYPES_IN_EXACT_ORDER_V1
        ],
        "schema_version": SCHEMA_VERSION_V1,
        "status": STATUS_V1,
        "claim_level": CLAIM_LEVEL_V1,
        "disposition": (
            R01PresentedMechanicsDispositionV1
            .VERIFIED_PRESENTED_R01_MECHANICS_NOT_R01
            .value
        ),
        "reason": (
            R01PresentedMechanicsReasonV1
            .EXACT_24_SOURCE_AND_FULL_RAW_LEAF_GRAPH_VERIFIED
            .value
        ),
        "issuer_callable": (
            "verify_phase2b_actual_unsealed_960_r01_real_current_cut_active_"
            "nonrevoked_presented_mechanics_v1"
        ),
        "require_callable": (
            "require_verified_phase2b_actual_unsealed_960_r01_presented_"
            "mechanics_v1"
        ),
        "issuer_only": True,
        "copy_deepcopy_pickle_or_mutated_result_acceptance_allowed": False,
        "mechanics_fields_required_exact_true": True,
        "nonclaim_fields_required_exact_false": True,
    },
    "public_presented_verifier": (
        "verify_phase2b_actual_unsealed_960_r01_real_current_cut_active_"
        "nonrevoked_presented_mechanics_v1"
    ),
    "public_real_verifier": (
        "verify_real_phase2b_actual_unsealed_960_r01_current_cut_active_"
        "nonrevoked_v1"
    ),
    "real_verification_status": REAL_VERIFICATION_STATUS_V1,
}
CONTRACT_ID_V1: Final = CONTRACT_ID_PREFIX_V1 + hashlib.sha256(
    CONTRACT_ID_DOMAIN_V1
    + canonical_phase2b_actual_unsealed_960_r01_presented_mechanics_bytes_v1(
        CONTRACT_BODY_V1
    )
).hexdigest()
CONTRACT_PAYLOAD_V1: Final = (
    canonical_phase2b_actual_unsealed_960_r01_presented_mechanics_bytes_v1(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION_V1,
            "body": CONTRACT_BODY_V1,
            "contract_id": CONTRACT_ID_V1,
        }
    )
)
CONTRACT_PAYLOAD_SHA256_V1: Final = hashlib.sha256(CONTRACT_PAYLOAD_V1).hexdigest()


def _build_private_boundary_v1():
    error_type = Phase2BActualUnsealed960R01PresentedMechanicsErrorV1
    result_type = VerifiedPhase2BActualUnsealed960R01PresentedMechanicsV1
    disposition_type = R01PresentedMechanicsDispositionV1
    reason_type = R01PresentedMechanicsReasonV1
    canonicalize = _CAPTURED_CANONICALIZE_V1
    decode_canonical = _CAPTURED_DECODE_V1
    sha256 = hashlib.sha256
    expected_catalog_payload = bytes(_catalog.CATALOG_PAYLOAD_V1)
    expected_catalog_id = str(_catalog.CATALOG_ID_V1)
    expected_catalog_digest = sha256(expected_catalog_payload).digest()
    expected_catalog_sha_text = "sha256:" + expected_catalog_digest.hex()
    source_kinds = tuple(R01_NATIVE_SOURCE_KINDS_IN_EXACT_ORDER_V1)
    predecessor_plan = tuple(
        tuple(row)
        for row in R01_NATIVE_SOURCE_PREDECESSOR_ORDINALS_IN_EXACT_ORDER_V1
    )
    state_profiles = tuple(R01_PRESENTED_STATE_PROFILES_IN_EXACT_ORDER_V1)
    composite_leaf_kinds = {
        ordinal: tuple(kinds)
        for ordinal, kinds in COMPOSITE_LEAF_KINDS_BY_SOURCE_ORDINAL_V1.items()
    }
    composite_ordinals = tuple(COMPOSITE_SOURCE_ORDINALS_IN_EXACT_ORDER_V1)
    expected_nonclaims = dict(NONCLAIM_ORACLE_V1)
    nonclaim_names = tuple(NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1)
    mechanics_names = tuple(MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1)
    native_envelope_fields = tuple(NATIVE_SOURCE_ENVELOPE_FIELDS_IN_EXACT_ORDER_V1)
    native_body_fields = tuple(NATIVE_SOURCE_BODY_FIELDS_IN_EXACT_ORDER_V1)
    leaf_envelope_fields = tuple(
        RAW_COMPOSITE_LEAF_ENVELOPE_FIELDS_IN_EXACT_ORDER_V1
    )
    leaf_body_fields = tuple(RAW_COMPOSITE_LEAF_BODY_FIELDS_IN_EXACT_ORDER_V1)
    manifest_fields = tuple(COMPOSITE_MANIFEST_FIELDS_IN_EXACT_ORDER_V1)
    maximum_bytes = MAXIMUM_EXACT_BYTES_PARAMETER_OR_TUPLE_ITEM_BYTES_V1
    result_fields = tuple(RESULT_FIELDS_IN_EXACT_ORDER_V1)
    contract_id = str(CONTRACT_ID_V1)
    type_builtin = type
    bytes_type = bytes
    tuple_type = tuple
    list_type = list
    dict_type = dict
    str_type = str
    int_type = int
    float_type = float
    bool_type = bool
    len_builtin = len
    all_builtin = all
    any_builtin = any
    tuple_builtin = tuple
    enumerate_builtin = enumerate
    zip_builtin = zip
    sum_builtin = sum
    list_builtin = list
    set_builtin = set
    abs_builtin = abs
    str_builtin = str
    id_builtin = id
    getattr_builtin = getattr
    exception_type = Exception
    object_new = object.__new__
    object_setattr = object.__setattr__
    issued: dict[int, tuple[object, tuple[object, ...]]] = {}
    _p_fail_type = FAIL_TYPE_V1
    _p_fail_catalog = FAIL_CATALOG_V1
    _p_fail_canonical = FAIL_CANONICAL_V1
    _p_fail_schema = FAIL_SCHEMA_V1
    _p_fail_content_id = FAIL_CONTENT_ID_V1
    _p_fail_binding = FAIL_BINDING_V1
    _p_fail_composite = FAIL_COMPOSITE_V1
    _p_fail_claim_boundary = FAIL_CLAIM_BOUNDARY_V1
    _p_fail_result = FAIL_RESULT_V1
    _p_fail_not_implemented = FAIL_NOT_IMPLEMENTED_V1
    _p_max_safe_integer = MAXIMUM_JSON_SAFE_INTEGER_V1
    _p_native_source_id_domain_prefix = NATIVE_SOURCE_ID_DOMAIN_PREFIX_V1
    _p_raw_leaf_id_domain_prefix = RAW_COMPOSITE_LEAF_ID_DOMAIN_PREFIX_V1
    _p_composite_root_domain_prefix = COMPOSITE_LEAF_ROOT_DOMAIN_PREFIX_V1
    _p_source_graph_root_domain = SOURCE_GRAPH_ROOT_DOMAIN_V1
    _p_native_envelope_schema = NATIVE_SOURCE_ENVELOPE_SCHEMA_VERSION_V1
    _p_native_body_schema = NATIVE_SOURCE_BODY_SCHEMA_VERSION_V1
    _p_native_status = NATIVE_SOURCE_STATUS_V1
    _p_native_claim_level = NATIVE_SOURCE_CLAIM_LEVEL_V1
    _p_leaf_envelope_schema = RAW_COMPOSITE_LEAF_ENVELOPE_SCHEMA_VERSION_V1
    _p_leaf_body_schema = RAW_COMPOSITE_LEAF_BODY_SCHEMA_VERSION_V1
    _p_leaf_status = RAW_LEAF_STATUS_V1
    _p_leaf_claim_level = RAW_LEAF_CLAIM_LEVEL_V1
    _p_manifest_schema = COMPOSITE_MANIFEST_SCHEMA_VERSION_V1
    _p_native_id_prefix = NATIVE_SOURCE_ID_PREFIX_V1
    _p_leaf_id_prefix = RAW_COMPOSITE_LEAF_ID_PREFIX_V1
    _p_gate_ordinal = CATALOG_R01_GATE_ORDINAL_V1
    _p_gate_id = CATALOG_R01_GATE_ID_V1
    _p_dispatch_id = CATALOG_R01_DISPATCH_ID_V1
    _p_result_schema = SCHEMA_VERSION_V1
    _p_result_status = STATUS_V1
    _p_result_claim_level = CLAIM_LEVEL_V1
    _p_total_raw_leaf_count = TOTAL_REQUIRED_RAW_COMPOSITE_LEAF_COUNT_V1
    isfinite = math.isfinite

    def restore_wire_integers(value: object) -> object:
        if type_builtin(value) is float_type:
            if (
                isfinite(value)
                and value.is_integer()
                and abs_builtin(value) <= _p_max_safe_integer
            ):
                return int_type(value)
            return value
        if type_builtin(value) is list_type:
            return list_type(restore_wire_integers(item) for item in value)
        if type_builtin(value) is dict_type:
            return dict_type(
                (key, restore_wire_integers(item))
                for key, item in value.items()
            )
        return value

    # PRIVATE_RUNTIME_CAPTURE_START: runtime code below uses only these aliases.

    def fail(code: str, detail: str) -> NoReturn:
        raise error_type(code, detail)

    def sha_address(payload: bytes) -> str:
        return "sha256:" + sha256(payload).hexdigest()

    def require_raw(payload: object, label: str) -> bytes:
        if type_builtin(payload) is not bytes_type:
            fail(_p_fail_type, f"{label} must be exact builtin bytes")
        if not payload or len_builtin(payload) > maximum_bytes:
            fail(_p_fail_type, f"{label} is empty or exceeds byte bound")
        if payload.endswith(b"\n"):
            fail(_p_fail_canonical, f"{label} has trailing LF")
        return payload

    def decode_exact(payload: object, label: str) -> dict[str, object]:
        raw = require_raw(payload, label)
        try:
            value = restore_wire_integers(decode_canonical(raw))
        except exception_type as exc:
            raise error_type(
                _p_fail_canonical, f"{label} is not exact RFC8785"
            ) from exc
        if type_builtin(value) is not dict_type or canonicalize(value) != raw:
            fail(_p_fail_canonical, f"{label} canonical round trip differs")
        return value

    def exact_object(
        value: object, fields: tuple[str, ...], label: str
    ) -> dict[str, object]:
        if (
            type_builtin(value) is not dict_type
            or set_builtin(value) != set_builtin(fields)
        ):
            fail(_p_fail_schema, f"{label} keyset differs")
        return value

    def require_text(value: object, label: str) -> str:
        if (
            type_builtin(value) is not str_type
            or not value
            or len_builtin(value) > 4096
        ):
            fail(_p_fail_schema, f"{label} must be exact nonempty text")
        return value

    def require_safe_int(value: object, label: str) -> int:
        if (
            type_builtin(value) is not int_type
            or value < 0
            or value > _p_max_safe_integer
        ):
            fail(_p_fail_schema, f"{label} must be an exact safe integer")
        return value

    def require_string_list(value: object, label: str) -> tuple[str, ...]:
        if type_builtin(value) is not list_type or any_builtin(
            type_builtin(item) is not str_type for item in value
        ):
            fail(_p_fail_schema, f"{label} must be an exact string array")
        return tuple_builtin(value)

    def require_int_list(value: object, label: str) -> tuple[int, ...]:
        if type_builtin(value) is not list_type:
            fail(_p_fail_schema, f"{label} must be an exact integer array")
        return tuple_builtin(require_safe_int(item, label) for item in value)

    def require_nonclaims(value: object, label: str) -> None:
        if (
            type_builtin(value) is not dict_type
            or set_builtin(value) != set_builtin(nonclaim_names)
            or len_builtin(value) != len_builtin(nonclaim_names)
        ):
            fail(_p_fail_claim_boundary, f"{label} nonclaim fields differ")
        for name in nonclaim_names:
            if type_builtin(value[name]) is not bool_type or value[name] is not False:
                fail(
                    _p_fail_claim_boundary,
                    f"{label}.{name} must be exact false",
                )

    def source_domain(ordinal: int) -> bytes:
        return (
            _p_native_source_id_domain_prefix
            + str_builtin(ordinal).encode("ascii")
            + b"/ID/V1\0"
        )

    def leaf_domain(source_ordinal: int, leaf_ordinal: int) -> bytes:
        return (
            _p_raw_leaf_id_domain_prefix
            + str_builtin(source_ordinal).encode("ascii")
            + b"/"
            + str_builtin(leaf_ordinal).encode("ascii")
            + b"/ID/V1\0"
        )

    def leaf_root_domain(source_ordinal: int) -> bytes:
        return (
            _p_composite_root_domain_prefix
            + str_builtin(source_ordinal).encode("ascii")
            + b"/V1\0"
        )

    def content_id(prefix: str, domain: bytes, body: object) -> str:
        return prefix + sha256(domain + canonicalize(body)).hexdigest()

    def load_leaf(
        raw: bytes,
        source_ordinal: int,
        source_kind: str,
        leaf_ordinal: int,
        leaf_kind: str,
    ) -> tuple[str, str, int]:
        value = exact_object(
            decode_exact(raw, f"source {source_ordinal} leaf {leaf_ordinal}"),
            leaf_envelope_fields,
            f"source {source_ordinal} leaf {leaf_ordinal} envelope",
        )
        if value["schema_version"] != _p_leaf_envelope_schema:
            fail(_p_fail_schema, "raw leaf envelope schema differs")
        body = exact_object(
            value["body"],
            leaf_body_fields,
            f"source {source_ordinal} leaf {leaf_ordinal} body",
        )
        if (
            body["schema_version"] != _p_leaf_body_schema
            or body["status"] != _p_leaf_status
            or body["claim_level"] != _p_leaf_claim_level
            or require_safe_int(
                body["parent_source_ordinal"], "raw leaf parent source ordinal"
            )
            != source_ordinal
            or body["parent_source_kind"] != source_kind
            or require_safe_int(body["leaf_ordinal"], "raw leaf ordinal")
            != leaf_ordinal
            or body["leaf_kind"] != leaf_kind
        ):
            fail(_p_fail_composite, "raw leaf descriptor differs")
        if body["payload_label"] != leaf_kind:
            fail(_p_fail_composite, "raw leaf payload label differs")
        require_nonclaims(body["nonclaims"], "raw leaf")
        expected_id = content_id(
            _p_leaf_id_prefix,
            leaf_domain(source_ordinal, leaf_ordinal),
            body,
        )
        if type_builtin(value["raw_leaf_id"]) is not str_type or value["raw_leaf_id"] != expected_id:
            fail(_p_fail_content_id, "raw leaf content ID differs")
        return expected_id, sha_address(raw), len_builtin(raw)

    def load_runtime_leaf(
        raw: bytes,
        expected_raw: bytes,
        expected_source_id: str,
        leaf_ordinal: int,
    ) -> tuple[str, str, int]:
        if raw != expected_raw:
            fail(
                _p_fail_composite,
                f"runtime bundle leaf {leaf_ordinal} is not byte-equal to source",
            )
        return expected_source_id, sha_address(raw), len_builtin(raw)

    def require_manifest(
        value: object,
        source_ordinal: int,
        source_kind: str,
        raw_leaves: tuple[bytes, ...],
        source_raws: tuple[bytes, ...],
        source_ids: tuple[str, ...],
    ) -> str:
        manifest = exact_object(
            value, manifest_fields, f"source {source_ordinal} composite manifest"
        )
        expected_kinds = composite_leaf_kinds[source_ordinal]
        if (
            manifest["schema_version"] != _p_manifest_schema
            or manifest["profile_id"]
            != f"R01_SOURCE_{source_ordinal:02d}_FULL_RAW_LEAVES_V1"
            or require_string_list(
                manifest["leaf_kinds_in_exact_order"], "manifest leaf kinds"
            )
            != expected_kinds
            or type_builtin(raw_leaves) is not tuple_type
            or len_builtin(raw_leaves) != len_builtin(expected_kinds)
        ):
            fail(_p_fail_composite, "composite profile or leaf cardinality differs")
        ids: list[str] = []
        shas: list[str] = []
        counts: list[int] = []
        for leaf_ordinal, (leaf_kind, raw_leaf) in enumerate_builtin(
            zip_builtin(expected_kinds, raw_leaves, strict=True)
        ):
            require_raw(raw_leaf, f"source {source_ordinal} leaf {leaf_ordinal}")
            if source_ordinal == 16:
                leaf_id, leaf_sha, leaf_count = load_runtime_leaf(
                    raw_leaf,
                    source_raws[11 + leaf_ordinal],
                    source_ids[11 + leaf_ordinal],
                    leaf_ordinal,
                )
            else:
                leaf_id, leaf_sha, leaf_count = load_leaf(
                    raw_leaf,
                    source_ordinal,
                    source_kind,
                    leaf_ordinal,
                    leaf_kind,
                )
            ids.append(leaf_id)
            shas.append(leaf_sha)
            counts.append(leaf_count)
        if (
            require_string_list(manifest["leaf_ids_in_exact_order"], "manifest leaf IDs")
            != tuple_builtin(ids)
            or require_string_list(
                manifest["leaf_payload_sha256s_in_exact_order"],
                "manifest leaf payload SHAs",
            )
            != tuple_builtin(shas)
            or require_int_list(
                manifest["leaf_payload_byte_counts_in_exact_order"],
                "manifest leaf byte counts",
            )
            != tuple_builtin(counts)
        ):
            fail(_p_fail_composite, "composite raw leaf binding differs")
        root_preimage = {
            "leaf_kinds_in_exact_order": list_builtin(expected_kinds),
            "leaf_ids_in_exact_order": ids,
            "leaf_payload_sha256s_in_exact_order": shas,
            "leaf_payload_byte_counts_in_exact_order": counts,
        }
        expected_root = sha_address(
            leaf_root_domain(source_ordinal) + canonicalize(root_preimage)
        )
        if manifest["leaf_root_sha256"] != expected_root:
            fail(_p_fail_composite, "composite raw leaf root differs")
        return expected_root

    def exact_equal(observed: object, expected: object) -> bool:
        if type_builtin(observed) is not type_builtin(expected):
            return False
        if type_builtin(expected) is tuple_type:
            return len_builtin(observed) == len_builtin(expected) and all_builtin(
                exact_equal(observed_item, expected_item)
                for observed_item, expected_item in zip_builtin(
                    observed, expected, strict=True
                )
            )
        if type_builtin(expected) is list_type:
            return len_builtin(observed) == len_builtin(expected) and all_builtin(
                exact_equal(observed_item, expected_item)
                for observed_item, expected_item in zip_builtin(
                    observed, expected, strict=True
                )
            )
        if type_builtin(expected) is dict_type:
            return (
                set_builtin(observed) == set_builtin(expected)
                and len_builtin(observed) == len_builtin(expected)
                and all_builtin(
                    exact_equal(observed[key], expected[key]) for key in expected
                )
            )
        return observed == expected

    def require_result(
        value: object,
    ) -> VerifiedPhase2BActualUnsealed960R01PresentedMechanicsV1:
        if type_builtin(value) is not result_type:
            fail(_p_fail_result, "private R01 mechanics result required")
        record = issued.get(id_builtin(value))
        if record is None or record[0] is not value:
            fail(_p_fail_result, "R01 mechanics result lacks private lineage")
        snapshot = tuple_builtin(
            getattr_builtin(value, field) for field in result_fields
        )
        if not exact_equal(snapshot, record[1]):
            fail(_p_fail_result, "R01 mechanics result was mutated")
        for name in mechanics_names:
            observed = getattr_builtin(value, name)
            if type_builtin(observed) is not bool_type or observed is not True:
                fail(_p_fail_result, f"mechanics result {name} differs")
        for name in nonclaim_names:
            observed = getattr_builtin(value, name)
            if type_builtin(observed) is not bool_type or observed is not False:
                fail(_p_fail_result, f"nonclaim result {name} differs")
        return value

    def issue(values: dict[str, object]):
        if tuple_builtin(values) != result_fields:
            fail(_p_fail_result, "private result field order differs")
        result = object_new(result_type)
        for field in result_fields:
            object_setattr(result, field, values[field])
        snapshot = tuple_builtin(
            getattr_builtin(result, field) for field in result_fields
        )
        issued[id_builtin(result)] = (result, snapshot)
        return require_result(result)

    def verify_presented(
        catalog_payload: bytes,
        raw_native_source_payloads_in_exact_order: tuple[bytes, ...],
        raw_composite_leaf_payload_tuples_in_exact_native_source_order: tuple[
            tuple[bytes, ...], ...
        ],
        *,
        out_of_band_expected_catalog_id_bytes: bytes,
        out_of_band_expected_catalog_payload_sha256_digest: bytes,
    ) -> VerifiedPhase2BActualUnsealed960R01PresentedMechanicsV1:
        require_raw(catalog_payload, "catalog payload")
        if (
            type_builtin(out_of_band_expected_catalog_id_bytes) is not bytes_type
            or type_builtin(out_of_band_expected_catalog_payload_sha256_digest)
            is not bytes_type
            or out_of_band_expected_catalog_id_bytes
            != expected_catalog_id.encode("ascii")
            or out_of_band_expected_catalog_payload_sha256_digest
            != expected_catalog_digest
            or catalog_payload != expected_catalog_payload
        ):
            fail(_p_fail_catalog, "catalog payload or independent pins differ")
        if (
            type_builtin(raw_native_source_payloads_in_exact_order)
            is not tuple_type
            or len_builtin(raw_native_source_payloads_in_exact_order) != 24
            or any_builtin(
                type_builtin(item) is not bytes_type
                for item in raw_native_source_payloads_in_exact_order
            )
        ):
            fail(_p_fail_type, "native sources must be one exact 24-byte tuple")
        if (
            type_builtin(
                raw_composite_leaf_payload_tuples_in_exact_native_source_order
            )
            is not tuple_type
            or len_builtin(
                raw_composite_leaf_payload_tuples_in_exact_native_source_order
            )
            != 24
            or any_builtin(
                type_builtin(row) is not tuple_type
                or any_builtin(
                    type_builtin(item) is not bytes_type for item in row
                )
                for row in raw_composite_leaf_payload_tuples_in_exact_native_source_order
            )
        ):
            fail(_p_fail_type, "composite leaf matrix must be exact 24 tuple rows")

        source_raws = tuple_builtin(raw_native_source_payloads_in_exact_order)
        leaf_matrix = tuple_builtin(
            tuple_builtin(row)
            for row in raw_composite_leaf_payload_tuples_in_exact_native_source_order
        )
        source_ids: list[str] = []
        source_shas: list[str] = []
        composite_roots: list[str] = []
        for ordinal, (source_kind, raw_source) in enumerate_builtin(
            zip_builtin(source_kinds, source_raws, strict=True)
        ):
            value = exact_object(
                decode_exact(raw_source, f"native source {ordinal}"),
                native_envelope_fields,
                f"native source {ordinal} envelope",
            )
            if value["schema_version"] != _p_native_envelope_schema:
                fail(_p_fail_schema, f"native source {ordinal} envelope schema differs")
            body = exact_object(
                value["body"],
                native_body_fields,
                f"native source {ordinal} body",
            )
            predecessors = predecessor_plan[ordinal]
            if (
                body["schema_version"] != _p_native_body_schema
                or body["status"] != _p_native_status
                or body["claim_level"] != _p_native_claim_level
                or body["catalog_id"] != expected_catalog_id
                or body["catalog_payload_sha256"] != expected_catalog_sha_text
                or require_safe_int(body["gate_ordinal"], "source gate ordinal")
                != _p_gate_ordinal
                or body["gate_id"] != _p_gate_id
                or body["verifier_dispatch_id"] != _p_dispatch_id
                or require_safe_int(
                    body["native_source_ordinal"], "native source ordinal"
                )
                != ordinal
                or body["native_source_kind"] != source_kind
                or body["presented_state_profile"] != state_profiles[ordinal]
                or require_int_list(
                    body["predecessor_ordinals_in_exact_order"],
                    "source predecessor ordinals",
                )
                != predecessors
                or require_string_list(
                    body["predecessor_source_ids_in_exact_order"],
                    "source predecessor IDs",
                )
                != tuple_builtin(source_ids[index] for index in predecessors)
                or require_string_list(
                    body["predecessor_source_payload_sha256s_in_exact_order"],
                    "source predecessor SHAs",
                )
                != tuple_builtin(source_shas[index] for index in predecessors)
            ):
                fail(_p_fail_binding, f"native source {ordinal} descriptor differs")
            require_nonclaims(body["nonclaims"], f"native source {ordinal}")
            row = leaf_matrix[ordinal]
            if ordinal in composite_leaf_kinds:
                root = require_manifest(
                    body["composite_leaf_manifest_or_none"],
                    ordinal,
                    source_kind,
                    row,
                    source_raws,
                    tuple_builtin(source_ids),
                )
                composite_roots.append(root)
            elif body["composite_leaf_manifest_or_none"] is not None or row != ():
                fail(_p_fail_composite, f"atomic source {ordinal} has composite leaves")
            expected_source_id = content_id(
                _p_native_id_prefix,
                source_domain(ordinal),
                body,
            )
            if (
                type_builtin(value["native_source_id"]) is not str_type
                or value["native_source_id"] != expected_source_id
            ):
                fail(_p_fail_content_id, f"native source {ordinal} content ID differs")
            source_ids.append(expected_source_id)
            source_shas.append(sha_address(raw_source))

        if len_builtin(composite_roots) != len_builtin(composite_ordinals):
            fail(_p_fail_composite, "composite source cardinality differs")
        if (
            sum_builtin(len_builtin(row) for row in leaf_matrix)
            != _p_total_raw_leaf_count
        ):
            fail(_p_fail_composite, "total raw composite leaf count differs")
        graph_preimage = {
            "native_source_kinds_in_exact_order": list_builtin(source_kinds),
            "native_source_ids_in_exact_order": source_ids,
            "native_source_payload_sha256s_in_exact_order": source_shas,
            "composite_leaf_roots_in_exact_source_order": composite_roots,
        }
        graph_root = sha_address(
            _p_source_graph_root_domain + canonicalize(graph_preimage)
        )
        values: dict[str, object] = {
            "schema_version": _p_result_schema,
            "status": _p_result_status,
            "claim_level": _p_result_claim_level,
            "disposition": disposition_type.VERIFIED_PRESENTED_R01_MECHANICS_NOT_R01,
            "reason": reason_type.EXACT_24_SOURCE_AND_FULL_RAW_LEAF_GRAPH_VERIFIED,
            "contract_id": contract_id,
            "catalog_id": expected_catalog_id,
            "catalog_payload_sha256": expected_catalog_sha_text,
            "gate_ordinal": _p_gate_ordinal,
            "gate_id": _p_gate_id,
            "verifier_dispatch_id": _p_dispatch_id,
            "native_source_count": len_builtin(source_ids),
            "composite_source_count": len_builtin(composite_roots),
            "raw_composite_leaf_count": sum_builtin(
                len_builtin(row) for row in leaf_matrix
            ),
            "source_graph_root_sha256": graph_root,
            "native_source_ids_in_exact_order": tuple_builtin(source_ids),
            "native_source_payload_sha256s_in_exact_order": tuple_builtin(
                source_shas
            ),
        }
        values.update({name: True for name in mechanics_names})
        values.update(expected_nonclaims)
        return issue(values)

    def verify_real(
        catalog_payload: bytes,
        raw_native_source_payloads_in_exact_order: tuple[bytes, ...],
        raw_composite_leaf_payload_tuples_in_exact_native_source_order: tuple[
            tuple[bytes, ...], ...
        ],
        *,
        out_of_band_expected_catalog_id_bytes: bytes,
        out_of_band_expected_catalog_payload_sha256_digest: bytes,
    ) -> NoReturn:
        del (
            catalog_payload,
            raw_native_source_payloads_in_exact_order,
            raw_composite_leaf_payload_tuples_in_exact_native_source_order,
            out_of_band_expected_catalog_id_bytes,
            out_of_band_expected_catalog_payload_sha256_digest,
        )
        fail(
            _p_fail_not_implemented,
            "real R01 requires external currentness, authority, custody, durability, and non-revocation evidence",
        )

    verify_presented.__name__ = (
        "verify_phase2b_actual_unsealed_960_r01_real_current_cut_active_"
        "nonrevoked_presented_mechanics_v1"
    )
    require_result.__name__ = (
        "require_verified_phase2b_actual_unsealed_960_r01_presented_mechanics_v1"
    )
    verify_real.__name__ = (
        "verify_real_phase2b_actual_unsealed_960_r01_current_cut_active_"
        "nonrevoked_v1"
    )
    return verify_presented, require_result, verify_real


(
    verify_phase2b_actual_unsealed_960_r01_real_current_cut_active_nonrevoked_presented_mechanics_v1,
    require_verified_phase2b_actual_unsealed_960_r01_presented_mechanics_v1,
    verify_real_phase2b_actual_unsealed_960_r01_current_cut_active_nonrevoked_v1,
) = _build_private_boundary_v1()


def _assert_public_abi_v1() -> None:
    expected_positional = tuple(name for name, _ in POSITIONAL_PARAMETERS_IN_EXACT_ORDER_V1)
    expected_keyword = tuple(KEYWORD_ONLY_PARAMETERS_IN_EXACT_ORDER_V1)
    for function in (
        verify_phase2b_actual_unsealed_960_r01_real_current_cut_active_nonrevoked_presented_mechanics_v1,
        verify_real_phase2b_actual_unsealed_960_r01_current_cut_active_nonrevoked_v1,
    ):
        code = function.__code__
        if (
            code.co_posonlyargcount != 0
            or code.co_argcount != len(expected_positional)
            or code.co_kwonlyargcount != len(expected_keyword)
            or tuple(code.co_varnames[: code.co_argcount]) != expected_positional
            or tuple(
                code.co_varnames[
                    code.co_argcount : code.co_argcount + code.co_kwonlyargcount
                ]
            )
            != expected_keyword
            or code.co_flags & 0x0C
            or function.__defaults__ is not None
            or function.__kwdefaults__ is not None
        ):
            raise RuntimeError("R01 downstream public ABI differs")
    if tuple(
        VerifiedPhase2BActualUnsealed960R01PresentedMechanicsV1.__dataclass_fields__
    ) != RESULT_FIELDS_IN_EXACT_ORDER_V1:
        raise RuntimeError("R01 downstream result field order differs")
    if len(MAIN_CAS_LEAF_KINDS_IN_EXACT_ORDER_V1) != 176:
        raise RuntimeError("R01 downstream main CAS raw leaf count differs")
    if TOTAL_REQUIRED_RAW_COMPOSITE_LEAF_COUNT_V1 != 193:
        raise RuntimeError("R01 downstream total raw leaf count differs")


_assert_public_abi_v1()
del _assert_public_abi_v1
del _build_private_boundary_v1


__all__ = [
    "CANONICALIZATION_V1",
    "CATALOG_R01_AVAILABILITY_V1",
    "CATALOG_R01_DISPATCH_ID_V1",
    "CATALOG_R01_GATE_ID_V1",
    "CATALOG_R01_GATE_ORDINAL_V1",
    "CLAIM_LEVEL_V1",
    "COMPOSITE_LEAF_KINDS_BY_SOURCE_ORDINAL_V1",
    "COMPOSITE_LEAF_ROOT_DOMAIN_PREFIX_V1",
    "COMPOSITE_MANIFEST_FIELDS_IN_EXACT_ORDER_V1",
    "COMPOSITE_MANIFEST_SCHEMA_VERSION_V1",
    "COMPOSITE_SOURCE_ORDINALS_IN_EXACT_ORDER_V1",
    "CONTRACT_BODY_V1",
    "CONTRACT_ID_V1",
    "CONTRACT_PAYLOAD_SHA256_V1",
    "CONTRACT_PAYLOAD_V1",
    "CONTRACT_SCHEMA_VERSION_V1",
    "CURRENT_CUT_LOGICAL_TIME_LEAF_KINDS_IN_EXACT_ORDER_V1",
    "DUAL_SIGNED_R01_RECEIPT_LEAF_KINDS_IN_EXACT_ORDER_V1",
    "FAIL_BINDING_V1",
    "FAIL_CANONICAL_V1",
    "FAIL_CATALOG_V1",
    "FAIL_CLAIM_BOUNDARY_V1",
    "FAIL_COMPOSITE_V1",
    "FAIL_CONTENT_ID_V1",
    "FAIL_NOT_IMPLEMENTED_V1",
    "FAIL_RESULT_V1",
    "FAIL_SCHEMA_V1",
    "FAIL_TYPE_V1",
    "KEYWORD_ONLY_PARAMETERS_IN_EXACT_ORDER_V1",
    "MAIN_CAS_LEAF_KINDS_IN_EXACT_ORDER_V1",
    "MAIN_CAS_OBJECT_COUNT_V1",
    "MAXIMUM_EXACT_BYTES_PARAMETER_OR_TUPLE_ITEM_BYTES_V1",
    "MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1",
    "NATIVE_SOURCE_BODY_SCHEMA_VERSION_V1",
    "NATIVE_SOURCE_BODY_FIELDS_IN_EXACT_ORDER_V1",
    "NATIVE_SOURCE_ENVELOPE_SCHEMA_VERSION_V1",
    "NATIVE_SOURCE_ENVELOPE_FIELDS_IN_EXACT_ORDER_V1",
    "NATIVE_SOURCE_ID_DOMAIN_PREFIX_V1",
    "NATIVE_SOURCE_ID_PREFIX_V1",
    "NATIVE_SOURCE_STATUS_V1",
    "NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1",
    "NONCLAIM_ORACLE_V1",
    "OWNER_STATUS_LEDGER_LEAF_KINDS_IN_EXACT_ORDER_V1",
    "POSITIONAL_PARAMETERS_IN_EXACT_ORDER_V1",
    "R01_NATIVE_SOURCE_KINDS_IN_EXACT_ORDER_V1",
    "R01_NATIVE_SOURCE_PREDECESSOR_ORDINALS_IN_EXACT_ORDER_V1",
    "R01_PRESENTED_STATE_PROFILES_IN_EXACT_ORDER_V1",
    "RAW_COMPOSITE_LEAF_BODY_SCHEMA_VERSION_V1",
    "RAW_COMPOSITE_LEAF_BODY_FIELDS_IN_EXACT_ORDER_V1",
    "RAW_COMPOSITE_LEAF_ENVELOPE_SCHEMA_VERSION_V1",
    "RAW_COMPOSITE_LEAF_ENVELOPE_FIELDS_IN_EXACT_ORDER_V1",
    "RAW_COMPOSITE_LEAF_ID_DOMAIN_PREFIX_V1",
    "RAW_COMPOSITE_LEAF_ID_PREFIX_V1",
    "RAW_LEAF_CLAIM_LEVEL_V1",
    "RAW_LEAF_STATUS_V1",
    "REAL_VERIFICATION_STATUS_V1",
    "RESULT_FIELD_TYPES_IN_EXACT_ORDER_V1",
    "RESULT_FIELDS_IN_EXACT_ORDER_V1",
    "RUNTIME_CLOSURE_BUNDLE_LEAF_KINDS_IN_EXACT_ORDER_V1",
    "SOURCE_GRAPH_ROOT_DOMAIN_V1",
    "SCHEMA_VERSION_V1",
    "STATUS_V1",
    "TOTAL_REQUIRED_RAW_COMPOSITE_LEAF_COUNT_V1",
    "Phase2BActualUnsealed960R01PresentedMechanicsErrorV1",
    "R01PresentedMechanicsDispositionV1",
    "R01PresentedMechanicsReasonV1",
    "VerifiedPhase2BActualUnsealed960R01PresentedMechanicsV1",
    "canonical_phase2b_actual_unsealed_960_r01_presented_mechanics_bytes_v1",
    "require_verified_phase2b_actual_unsealed_960_r01_presented_mechanics_v1",
    "verify_phase2b_actual_unsealed_960_r01_real_current_cut_active_nonrevoked_presented_mechanics_v1",
    "verify_real_phase2b_actual_unsealed_960_r01_current_cut_active_nonrevoked_v1",
]
