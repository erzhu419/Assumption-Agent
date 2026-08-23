"""Independent KATs for the downstream R01 presented-mechanics verifier.

The corpus below is synthetic, deterministic, and TEST_ONLY.  It deliberately
uses an oracle copied as literals instead of deriving the 24-source order from
the implementation or catalog module.  Acceptance proves only that the raw
envelope/leaf mechanics presented to this local verifier are self-consistent.
It does not prove currentness, authority, owner non-revocation, R01, readiness,
execution, an actual-960 run, or C1.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Callable

import pytest

import hegel_machine.phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1 as catalog
import hegel_machine.phase2b_actual_unsealed_960_verify_r01_real_current_cut_active_nonrevoked_v1 as mechanics
import hegel_machine.rfc8785_v1 as rfc8785_module
from hegel_machine.rfc8785_v1 import decode_and_require_canonical_rfc8785_v1


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "src"
    / "hegel_machine"
    / "phase2b_actual_unsealed_960_verify_r01_real_current_cut_active_nonrevoked_v1.py"
)
CONFIG = (
    ROOT
    / "config"
    / "phase2b_actual_unsealed_960_verify_r01_real_current_cut_active_nonrevoked_v1.json"
)

CATALOG_ID = (
    "phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1_"
    "a9f1b10bf7df69065f9847a69b935613ba1c368832c21ea5ac60e863961272cd"
)
CATALOG_SHA256 = "62f46cfba8a86f7ad969a7ea584c1828f0b34bb1d95571b1cc64259ac01a588a"
SCHEMA_VERSION = (
    "hegel-machine-phase2b-actual-unsealed-960-verify-r01-real-current-cut-"
    "active-nonrevoked-presented-mechanics/1"
)
STATUS = "VERIFIED_PRESENTED_R01_24_SOURCE_FULL_RAW_LEAF_MECHANICS_NOT_R01"
CLAIM_LEVEL = (
    "LOCAL_OFFLINE_PRODUCER_FREE_PRESENTED_MECHANICS_ONLY_NOT_CURRENTNESS_"
    "NOT_AUTHORITY_NOT_OWNER_NONREVOCATION_NOT_R01_NOT_READINESS_NOT_"
    "EXECUTION_NOT_ACTUAL960_NOT_C1"
)
DISPATCH_ID = "VERIFY_R01_REAL_CURRENT_CUT_ACTIVE_NONREVOKED_V1"

# Independent literal oracle.  Do not replace this with a projection from the
# implementation or catalog: this tuple is the principal catalog-order KAT.
SOURCE_KINDS = (
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
PREDECESSOR_ORDINALS = (
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
STATE_PROFILES = (
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

OWNER_LEAF_KINDS = (
    "OWNER_STATUS_GENESIS_HEAD",
    "OWNER_STATUS_ACTIVE_DECISION",
    "OWNER_STATUS_ACTIVE_SIGNED_EVENT",
    "OWNER_STATUS_ACTIVE_CHECKPOINT",
    "OWNER_STATUS_ACTIVE_HEAD",
    "OWNER_STATUS_ACTIVE_PROJECTION",
)
TIME_LEAF_KINDS = (
    "CURRENT_CUT_LOGICAL_TIME_GENESIS_HEAD",
    "CURRENT_CUT_LOGICAL_TIME_REQUEST",
    "CURRENT_CUT_LOGICAL_TIME_SIGNED_TOKEN",
    "CURRENT_CUT_LOGICAL_TIME_CHECKPOINT",
    "CURRENT_CUT_LOGICAL_TIME_CURRENT_HEAD",
)
RUNTIME_LEAF_KINDS = tuple(
    f"LIVE_RUNTIME_CLOSURE_REPORT_{index}" for index in range(5)
)
CAS_LEAF_KINDS = (
    ("CAS_RETENTION_PLAN",)
    + tuple(f"CAS_REQUEST_{index:03d}" for index in range(29))
    + tuple(f"CAS_OBJECT_{index:03d}" for index in range(29))
    + tuple(f"CAS_SIGNED_RECEIPT_{index:03d}" for index in range(29))
    + tuple(f"CAS_CHECKPOINT_{index:03d}" for index in range(29))
    + tuple(f"CAS_HEAD_{index:03d}" for index in range(30))
    + tuple(f"CAS_RESPONSE_{index:03d}" for index in range(29))
)
DUAL_RECEIPT_LEAF_KINDS = ("R01_RECEIPT_BODY",)
COMPOSITE_LEAVES = {
    4: OWNER_LEAF_KINDS,
    7: TIME_LEAF_KINDS,
    16: RUNTIME_LEAF_KINDS,
    22: CAS_LEAF_KINDS,
    23: DUAL_RECEIPT_LEAF_KINDS,
}

NONCLAIMS = (
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
MECHANICS = (
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
RESULT_FIELDS = (
    "schema_version",
    "status",
    "claim_level",
    "disposition",
    "reason",
    "contract_id",
    "catalog_id",
    "catalog_payload_sha256",
    "gate_ordinal",
    "gate_id",
    "verifier_dispatch_id",
    "native_source_count",
    "composite_source_count",
    "raw_composite_leaf_count",
    "source_graph_root_sha256",
    "native_source_ids_in_exact_order",
    "native_source_payload_sha256s_in_exact_order",
    *MECHANICS,
    *NONCLAIMS,
)
RESULT_FIELD_TYPES = (
    ("schema_version", "str"),
    ("status", "str"),
    ("claim_level", "str"),
    ("disposition", "R01PresentedMechanicsDispositionV1"),
    ("reason", "R01PresentedMechanicsReasonV1"),
    ("contract_id", "str"),
    ("catalog_id", "str"),
    ("catalog_payload_sha256", "str"),
    ("gate_ordinal", "int"),
    ("gate_id", "str"),
    ("verifier_dispatch_id", "str"),
    ("native_source_count", "int"),
    ("composite_source_count", "int"),
    ("raw_composite_leaf_count", "int"),
    ("source_graph_root_sha256", "str"),
    ("native_source_ids_in_exact_order", "tuple[str, ...]"),
    ("native_source_payload_sha256s_in_exact_order", "tuple[str, ...]"),
    *((name, "bool") for name in (*MECHANICS, *NONCLAIMS)),
)
PUBLIC_SURFACE = (
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
)

NATIVE_ENVELOPE_SCHEMA = (
    "hegel-machine-phase2b-actual-unsealed-960-r01-native-source-envelope/1"
)
NATIVE_BODY_SCHEMA = (
    "hegel-machine-phase2b-actual-unsealed-960-r01-native-source-body/1"
)
LEAF_ENVELOPE_SCHEMA = (
    "hegel-machine-phase2b-actual-unsealed-960-r01-raw-composite-leaf/1"
)
LEAF_BODY_SCHEMA = (
    "hegel-machine-phase2b-actual-unsealed-960-r01-raw-composite-leaf-body/1"
)
MANIFEST_SCHEMA = (
    "hegel-machine-phase2b-actual-unsealed-960-r01-composite-manifest/1"
)
NATIVE_STATUS = "PRESENTED_MECHANICS_ONLY"
LEAF_STATUS = "PRESENTED_RAW_TERMINAL_LEAF_MECHANICS_ONLY"
NATIVE_ID_PREFIX = "phase2b_actual_unsealed_960_r01_native_source_v1_"
LEAF_ID_PREFIX = "phase2b_actual_unsealed_960_r01_raw_composite_leaf_v1_"
NATIVE_DOMAIN_PREFIX = b"HEGEL/PHASE2B/ACTUAL960/R01/NATIVE_SOURCE/"
LEAF_DOMAIN_PREFIX = b"HEGEL/PHASE2B/ACTUAL960/R01/RAW_COMPOSITE_LEAF/"
LEAF_ROOT_DOMAIN_PREFIX = b"HEGEL/PHASE2B/ACTUAL960/R01/COMPOSITE_LEAF_ROOT/"
GRAPH_ROOT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/R01/SOURCE_GRAPH_ROOT/V1\0"

# Filled from the independent length-framed corpus builder and intentionally
# frozen after the first focused run.
EXPECTED_CORPUS_FINGERPRINT_V1 = (
    "41a3219b57968272c89548e92b1f8c01256e3ede5e08c8a3106f4edc121165cd"
)
EXPECTED_SOURCE_GRAPH_ROOT_V1 = (
    "sha256:59727ae147c0035e62219603810c4e9845e5d5f9a9c2687e61a2b60fb0806f18"
)


class _BytesSubclass(bytes):
    pass


class _EqAll:
    """Adversarial equality object that must not bypass exact nested types."""

    def __eq__(self, other: object) -> bool:
        del other
        return True


@dataclass(frozen=True)
class _Corpus:
    positional: tuple[object, ...]
    keyword_only: dict[str, bytes]
    source_ids: tuple[str, ...]
    source_shas: tuple[str, ...]
    graph_root: str
    fingerprint: str


def _canonical(value: object) -> bytes:
    """Independent RFC8785 oracle for this ASCII/integer-only KAT domain."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _decoded(raw: bytes) -> dict[str, object]:
    value = json.loads(raw)
    assert type(value) is dict
    return value


def _sha_address(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _native_domain(ordinal: int) -> bytes:
    return NATIVE_DOMAIN_PREFIX + str(ordinal).encode("ascii") + b"/ID/V1\0"


def _leaf_domain(source_ordinal: int, leaf_ordinal: int) -> bytes:
    return (
        LEAF_DOMAIN_PREFIX
        + str(source_ordinal).encode("ascii")
        + b"/"
        + str(leaf_ordinal).encode("ascii")
        + b"/ID/V1\0"
    )


def _leaf_root_domain(source_ordinal: int) -> bytes:
    return (
        LEAF_ROOT_DOMAIN_PREFIX
        + str(source_ordinal).encode("ascii")
        + b"/V1\0"
    )


def _false_nonclaims() -> dict[str, bool]:
    return {name: False for name in NONCLAIMS}


def _make_leaf(
    source_ordinal: int,
    source_kind: str,
    leaf_ordinal: int,
    leaf_kind: str,
) -> tuple[bytes, str]:
    body = {
        "schema_version": LEAF_BODY_SCHEMA,
        "status": LEAF_STATUS,
        "claim_level": CLAIM_LEVEL,
        "parent_source_ordinal": source_ordinal,
        "parent_source_kind": source_kind,
        "leaf_ordinal": leaf_ordinal,
        "leaf_kind": leaf_kind,
        "payload_label": leaf_kind,
        "nonclaims": _false_nonclaims(),
    }
    leaf_id = LEAF_ID_PREFIX + hashlib.sha256(
        _leaf_domain(source_ordinal, leaf_ordinal) + _canonical(body)
    ).hexdigest()
    envelope = {
        "schema_version": LEAF_ENVELOPE_SCHEMA,
        "body": body,
        "raw_leaf_id": leaf_id,
    }
    return _canonical(envelope), leaf_id


def _make_manifest(
    source_ordinal: int,
    leaf_kinds: tuple[str, ...],
    leaf_raws: tuple[bytes, ...],
    leaf_ids: tuple[str, ...],
) -> tuple[dict[str, object], str]:
    shas = tuple(_sha_address(raw) for raw in leaf_raws)
    counts = tuple(len(raw) for raw in leaf_raws)
    preimage = {
        "leaf_kinds_in_exact_order": list(leaf_kinds),
        "leaf_ids_in_exact_order": list(leaf_ids),
        "leaf_payload_sha256s_in_exact_order": list(shas),
        "leaf_payload_byte_counts_in_exact_order": list(counts),
    }
    root = _sha_address(_leaf_root_domain(source_ordinal) + _canonical(preimage))
    return (
        {
            "schema_version": MANIFEST_SCHEMA,
            "profile_id": f"R01_SOURCE_{source_ordinal:02d}_FULL_RAW_LEAVES_V1",
            **preimage,
            "leaf_root_sha256": root,
        },
        root,
    )


def _build_corpus() -> _Corpus:
    catalog_payload = bytes(catalog.CATALOG_PAYLOAD_V1)
    assert hashlib.sha256(catalog_payload).hexdigest() == CATALOG_SHA256
    source_raws: list[bytes] = []
    source_ids: list[str] = []
    source_shas: list[str] = []
    leaf_matrix: list[tuple[bytes, ...]] = []
    composite_roots: list[str] = []

    for ordinal, source_kind in enumerate(SOURCE_KINDS):
        if ordinal in COMPOSITE_LEAVES and ordinal != 16:
            leaf_kinds = COMPOSITE_LEAVES[ordinal]
            built = tuple(
                _make_leaf(ordinal, source_kind, index, kind)
                for index, kind in enumerate(leaf_kinds)
            )
            leaf_raws = tuple(row[0] for row in built)
            leaf_ids = tuple(row[1] for row in built)
        elif ordinal == 16:
            leaf_kinds = RUNTIME_LEAF_KINDS
            leaf_raws = tuple(source_raws[11:16])
            leaf_ids = tuple(source_ids[11:16])
        else:
            leaf_kinds = ()
            leaf_raws = ()
            leaf_ids = ()

        if ordinal in COMPOSITE_LEAVES:
            manifest, composite_root = _make_manifest(
                ordinal, leaf_kinds, leaf_raws, leaf_ids
            )
            composite_roots.append(composite_root)
        else:
            manifest = None

        predecessors = PREDECESSOR_ORDINALS[ordinal]
        body = {
            "schema_version": NATIVE_BODY_SCHEMA,
            "status": NATIVE_STATUS,
            "claim_level": CLAIM_LEVEL,
            "catalog_id": CATALOG_ID,
            "catalog_payload_sha256": "sha256:" + CATALOG_SHA256,
            "gate_ordinal": 1,
            "gate_id": "R01_AUTHORIZATION_VERIFIED",
            "verifier_dispatch_id": DISPATCH_ID,
            "native_source_ordinal": ordinal,
            "native_source_kind": source_kind,
            "predecessor_ordinals_in_exact_order": list(predecessors),
            "predecessor_source_ids_in_exact_order": [
                source_ids[index] for index in predecessors
            ],
            "predecessor_source_payload_sha256s_in_exact_order": [
                source_shas[index] for index in predecessors
            ],
            "presented_state_profile": STATE_PROFILES[ordinal],
            "composite_leaf_manifest_or_none": manifest,
            "nonclaims": _false_nonclaims(),
        }
        source_id = NATIVE_ID_PREFIX + hashlib.sha256(
            _native_domain(ordinal) + _canonical(body)
        ).hexdigest()
        raw = _canonical(
            {
                "schema_version": NATIVE_ENVELOPE_SCHEMA,
                "body": body,
                "native_source_id": source_id,
            }
        )
        source_raws.append(raw)
        source_ids.append(source_id)
        source_shas.append(_sha_address(raw))
        leaf_matrix.append(leaf_raws)

    graph_preimage = {
        "native_source_kinds_in_exact_order": list(SOURCE_KINDS),
        "native_source_ids_in_exact_order": source_ids,
        "native_source_payload_sha256s_in_exact_order": source_shas,
        "composite_leaf_roots_in_exact_source_order": composite_roots,
    }
    graph_root = _sha_address(GRAPH_ROOT_DOMAIN + _canonical(graph_preimage))
    keyword_only = {
        "out_of_band_expected_catalog_id_bytes": CATALOG_ID.encode("ascii"),
        "out_of_band_expected_catalog_payload_sha256_digest": bytes.fromhex(
            CATALOG_SHA256
        ),
    }

    fingerprint = hashlib.sha256()
    fingerprint.update(b"HEGEL/TEST_ONLY/R01/PRESENTED_CORPUS/V1\0")
    for raw in source_raws:
        fingerprint.update(len(raw).to_bytes(8, "big"))
        fingerprint.update(raw)
    for row in leaf_matrix:
        fingerprint.update(len(row).to_bytes(4, "big"))
        for raw in row:
            fingerprint.update(len(raw).to_bytes(8, "big"))
            fingerprint.update(raw)
    for name, raw in keyword_only.items():
        fingerprint.update(name.encode("ascii") + b"\0")
        fingerprint.update(len(raw).to_bytes(8, "big"))
        fingerprint.update(raw)
    return _Corpus(
        positional=(catalog_payload, tuple(source_raws), tuple(leaf_matrix)),
        keyword_only=keyword_only,
        source_ids=tuple(source_ids),
        source_shas=tuple(source_shas),
        graph_root=graph_root,
        fingerprint=fingerprint.hexdigest(),
    )


@pytest.fixture(scope="session")
def corpus() -> _Corpus:
    return _build_corpus()


def _verify(
    corpus: _Corpus,
    *,
    positional: tuple[object, ...] | None = None,
    keyword_only: dict[str, object] | None = None,
) -> mechanics.VerifiedPhase2BActualUnsealed960R01PresentedMechanicsV1:
    return mechanics.verify_phase2b_actual_unsealed_960_r01_real_current_cut_active_nonrevoked_presented_mechanics_v1(
        *(corpus.positional if positional is None else positional),
        **(corpus.keyword_only if keyword_only is None else keyword_only),
    )


def _assert_error(code: str, operation: Callable[[], object]) -> None:
    with pytest.raises(
        mechanics.Phase2BActualUnsealed960R01PresentedMechanicsErrorV1
    ) as raised:
        operation()
    assert raised.value.code == code
    assert str(raised.value).startswith(code + ": ")


def _replace_positional(
    corpus: _Corpus, index: int, value: object
) -> tuple[object, ...]:
    values = list(corpus.positional)
    values[index] = value
    return tuple(values)


def _replace_source_raw(
    corpus: _Corpus, ordinal: int, raw: bytes
) -> tuple[object, ...]:
    raws = list(corpus.positional[1])
    raws[ordinal] = raw
    return _replace_positional(corpus, 1, tuple(raws))


def _replace_leaf_row(
    corpus: _Corpus, ordinal: int, row: object
) -> tuple[object, ...]:
    matrix = list(corpus.positional[2])
    matrix[ordinal] = row
    return _replace_positional(corpus, 2, tuple(matrix))


def _rewrite_source(
    raw: bytes,
    mutator: Callable[[dict[str, object]], None],
    *,
    rehash: bool = False,
) -> bytes:
    envelope = _decoded(raw)
    body = envelope["body"]
    assert type(body) is dict
    mutator(body)
    if rehash:
        ordinal = body["native_source_ordinal"]
        assert type(ordinal) is int
        envelope["native_source_id"] = NATIVE_ID_PREFIX + hashlib.sha256(
            _native_domain(ordinal) + _canonical(body)
        ).hexdigest()
    return _canonical(envelope)


def _rewrite_leaf(
    raw: bytes,
    source_ordinal: int,
    leaf_ordinal: int,
    mutator: Callable[[dict[str, object]], None],
    *,
    rehash: bool = False,
) -> bytes:
    envelope = _decoded(raw)
    body = envelope["body"]
    assert type(body) is dict
    mutator(body)
    if rehash:
        envelope["raw_leaf_id"] = LEAF_ID_PREFIX + hashlib.sha256(
            _leaf_domain(source_ordinal, leaf_ordinal) + _canonical(body)
        ).hexdigest()
    return _canonical(envelope)


def test_01_literal_catalog_order_composites_contract_and_public_abi() -> None:
    assert len(SOURCE_KINDS) == 24
    assert tuple(mechanics.R01_NATIVE_SOURCE_KINDS_IN_EXACT_ORDER_V1) == SOURCE_KINDS
    assert tuple(mechanics.R01_NATIVE_SOURCE_PREDECESSOR_ORDINALS_IN_EXACT_ORDER_V1) == PREDECESSOR_ORDINALS
    assert tuple(mechanics.R01_PRESENTED_STATE_PROFILES_IN_EXACT_ORDER_V1) == STATE_PROFILES
    assert tuple(mechanics.COMPOSITE_SOURCE_ORDINALS_IN_EXACT_ORDER_V1) == (
        4,
        7,
        16,
        22,
        23,
    )
    assert {
        ordinal: tuple(kinds)
        for ordinal, kinds in mechanics.COMPOSITE_LEAF_KINDS_BY_SOURCE_ORDINAL_V1.items()
    } == COMPOSITE_LEAVES
    assert tuple(mechanics.MAIN_CAS_LEAF_KINDS_IN_EXACT_ORDER_V1) == CAS_LEAF_KINDS
    assert len(CAS_LEAF_KINDS) == 1 + 29 * 4 + 30 + 29 == 176
    assert sum(map(len, COMPOSITE_LEAVES.values())) == 6 + 5 + 5 + 176 + 1 == 193
    assert mechanics.MAIN_CAS_OBJECT_COUNT_V1 == 29
    assert mechanics.TOTAL_REQUIRED_RAW_COMPOSITE_LEAF_COUNT_V1 == 193
    assert tuple(mechanics.NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1) == NONCLAIMS
    assert tuple(mechanics.MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1) == MECHANICS
    assert tuple(mechanics.RESULT_FIELDS_IN_EXACT_ORDER_V1) == RESULT_FIELDS
    assert tuple(mechanics.RESULT_FIELD_TYPES_IN_EXACT_ORDER_V1) == RESULT_FIELD_TYPES
    assert tuple(mechanics.__all__) == PUBLIC_SURFACE

    catalog_payload = decode_and_require_canonical_rfc8785_v1(catalog.CATALOG_PAYLOAD_V1)
    descriptor = catalog_payload["body"]["gate_source_schema_objects_in_exact_order"][0]
    assert descriptor["allowed_native_source_kind_orders_in_exact_order"] == [
        list(SOURCE_KINDS)
    ]
    assert descriptor["verifier_dispatch_id"] == DISPATCH_ID
    assert descriptor["verification_availability"] == "SCHEMA_FROZEN_NOT_VERIFIABLE"
    assert descriptor["verifier_source_sha256_or_none"] is None

    expected_names = tuple(
        name for name, _ in mechanics.POSITIONAL_PARAMETERS_IN_EXACT_ORDER_V1
    ) + tuple(mechanics.KEYWORD_ONLY_PARAMETERS_IN_EXACT_ORDER_V1)
    assert expected_names == (
        "catalog_payload",
        "raw_native_source_payloads_in_exact_order",
        "raw_composite_leaf_payload_tuples_in_exact_native_source_order",
        "out_of_band_expected_catalog_id_bytes",
        "out_of_band_expected_catalog_payload_sha256_digest",
    )
    for function in (
        mechanics.verify_phase2b_actual_unsealed_960_r01_real_current_cut_active_nonrevoked_presented_mechanics_v1,
        mechanics.verify_real_phase2b_actual_unsealed_960_r01_current_cut_active_nonrevoked_v1,
    ):
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == expected_names
        parameters = tuple(signature.parameters.values())
        assert all(
            item.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            for item in parameters[:3]
        )
        assert all(
            item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters[3:]
        )
        assert all(item.default is inspect.Parameter.empty for item in parameters)
    body = mechanics.CONTRACT_BODY_V1
    assert body["catalog_v1_modified_or_reissued"] is False
    assert body["catalog_dispatch_registration_performed"] is False
    assert body["catalog_r01_gate_success_allowed"] is False
    assert body["opaque_id_or_sha_only_composite_accepted"] is False
    assert body["runtime_bundle_leaves_must_equal_native_sources_11_through_15"] is True
    assert body["dual_receipt_requires_raw_receipt_body_leaf"] is True
    assert body["real_verification_status"] == "NOT_IMPLEMENTED"
    assert body["maximum_exact_bytes_parameter_or_tuple_item_bytes"] == 16_777_216

    native_contract = body["native_source_envelope_contract"]
    leaf_contract = body["raw_composite_leaf_envelope_contract"]
    manifest_contract = body["composite_manifest_contract"]
    graph_contract = body["source_graph_root_contract"]
    matrix_contract = body["raw_leaf_matrix_contract"]
    assert native_contract["envelope_fields_in_exact_order"] == [
        "schema_version",
        "body",
        "native_source_id",
    ]
    assert native_contract["body_fields_in_exact_order"] == [
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
    ]
    assert native_contract["content_id_prefix"] == NATIVE_ID_PREFIX
    assert native_contract["content_id_digest_preimage_formula"] == (
        "domain_bytes || RFC8785_EXACT_BYTES(body)"
    )
    assert leaf_contract["envelope_fields_in_exact_order"] == [
        "schema_version",
        "body",
        "raw_leaf_id",
    ]
    assert leaf_contract["body_fields_in_exact_order"] == [
        "schema_version",
        "status",
        "claim_level",
        "parent_source_ordinal",
        "parent_source_kind",
        "leaf_ordinal",
        "leaf_kind",
        "payload_label",
        "nonclaims",
    ]
    assert leaf_contract["payload_label_formula"] == "EXACT_LEAF_KIND_LITERAL"
    assert leaf_contract["content_id_prefix"] == LEAF_ID_PREFIX
    assert manifest_contract["fields_in_exact_order"] == [
        "schema_version",
        "profile_id",
        "leaf_kinds_in_exact_order",
        "leaf_ids_in_exact_order",
        "leaf_payload_sha256s_in_exact_order",
        "leaf_payload_byte_counts_in_exact_order",
        "leaf_root_sha256",
    ]
    assert manifest_contract["leaf_root_preimage_fields_in_exact_order"] == [
        "leaf_kinds_in_exact_order",
        "leaf_ids_in_exact_order",
        "leaf_payload_sha256s_in_exact_order",
        "leaf_payload_byte_counts_in_exact_order",
    ]
    assert graph_contract["preimage_fields_in_exact_order"] == [
        "native_source_kinds_in_exact_order",
        "native_source_ids_in_exact_order",
        "native_source_payload_sha256s_in_exact_order",
        "composite_leaf_roots_in_exact_source_order",
    ]
    assert matrix_contract == {
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
        "total_required_raw_leaves": 193,
    }
    assert body["private_result_contract"] == {
        "type_name": "VerifiedPhase2BActualUnsealed960R01PresentedMechanicsV1",
        "fields_and_types_in_exact_order": [list(row) for row in RESULT_FIELD_TYPES],
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "claim_level": CLAIM_LEVEL,
        "disposition": "VERIFIED_PRESENTED_R01_MECHANICS_NOT_R01",
        "reason": "EXACT_24_SOURCE_AND_FULL_RAW_LEAF_GRAPH_VERIFIED",
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
    }


def test_02_full_deterministic_positive_corpus_is_presented_mechanics_only(
    corpus: _Corpus,
) -> None:
    print("r01-presented-corpus-fingerprint", corpus.fingerprint)
    print("r01-presented-source-graph-root", corpus.graph_root)
    assert corpus.fingerprint == EXPECTED_CORPUS_FINGERPRINT_V1
    assert corpus.graph_root == EXPECTED_SOURCE_GRAPH_ROOT_V1
    assert _build_corpus() == corpus
    source_raws = corpus.positional[1]
    leaf_matrix = corpus.positional[2]
    assert type(source_raws) is tuple and len(source_raws) == 24
    assert type(leaf_matrix) is tuple and len(leaf_matrix) == 24
    assert tuple(len(row) for row in leaf_matrix) == (
        0,
        0,
        0,
        0,
        6,
        0,
        0,
        5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        5,
        0,
        0,
        0,
        0,
        0,
        176,
        1,
    )
    assert leaf_matrix[16] == source_raws[11:16]
    assert all(type(raw) is bytes for raw in source_raws)
    assert all(type(raw) is bytes for row in leaf_matrix for raw in row)
    assert all(_canonical(json.loads(raw)) == raw for raw in source_raws)
    assert all(_canonical(json.loads(raw)) == raw for row in leaf_matrix for raw in row)

    result = _verify(corpus)
    assert mechanics.require_verified_phase2b_actual_unsealed_960_r01_presented_mechanics_v1(result) is result
    assert result.schema_version == SCHEMA_VERSION
    assert result.status == STATUS
    assert result.claim_level == CLAIM_LEVEL
    assert result.disposition is mechanics.R01PresentedMechanicsDispositionV1.VERIFIED_PRESENTED_R01_MECHANICS_NOT_R01
    assert result.reason is mechanics.R01PresentedMechanicsReasonV1.EXACT_24_SOURCE_AND_FULL_RAW_LEAF_GRAPH_VERIFIED
    assert result.native_source_count == 24
    assert result.composite_source_count == 5
    assert result.raw_composite_leaf_count == 193
    assert result.source_graph_root_sha256 == corpus.graph_root
    assert result.native_source_ids_in_exact_order == corpus.source_ids
    assert result.native_source_payload_sha256s_in_exact_order == corpus.source_shas
    for field in MECHANICS:
        assert type(getattr(result, field)) is bool
        assert getattr(result, field) is True
    for field in NONCLAIMS:
        assert type(getattr(result, field)) is bool
        assert getattr(result, field) is False


@pytest.mark.parametrize(
    ("case", "expected_code"),
    (
        ("catalog-bytearray", mechanics.FAIL_TYPE_V1),
        ("catalog-subclass", mechanics.FAIL_TYPE_V1),
        ("source-list", mechanics.FAIL_TYPE_V1),
        ("source-item-subclass", mechanics.FAIL_TYPE_V1),
        ("matrix-list", mechanics.FAIL_TYPE_V1),
        ("matrix-row-list", mechanics.FAIL_TYPE_V1),
        ("leaf-item-subclass", mechanics.FAIL_TYPE_V1),
        ("catalog-id-pin-str", mechanics.FAIL_CATALOG_V1),
        ("catalog-sha-pin-bytearray", mechanics.FAIL_CATALOG_V1),
    ),
)
def test_03_exact_builtin_transport_types_reject_subclasses_and_lists(
    corpus: _Corpus, case: str, expected_code: str
) -> None:
    positional = corpus.positional
    keyword: dict[str, object] = dict(corpus.keyword_only)
    if case == "catalog-bytearray":
        positional = _replace_positional(corpus, 0, bytearray(corpus.positional[0]))
    elif case == "catalog-subclass":
        positional = _replace_positional(corpus, 0, _BytesSubclass(corpus.positional[0]))
    elif case == "source-list":
        positional = _replace_positional(corpus, 1, list(corpus.positional[1]))
    elif case == "source-item-subclass":
        raws = list(corpus.positional[1])
        raws[0] = _BytesSubclass(raws[0])
        positional = _replace_positional(corpus, 1, tuple(raws))
    elif case == "matrix-list":
        positional = _replace_positional(corpus, 2, list(corpus.positional[2]))
    elif case == "matrix-row-list":
        matrix = list(corpus.positional[2])
        matrix[4] = list(matrix[4])
        positional = _replace_positional(corpus, 2, tuple(matrix))
    elif case == "leaf-item-subclass":
        matrix = list(corpus.positional[2])
        row = list(matrix[4])
        row[0] = _BytesSubclass(row[0])
        matrix[4] = tuple(row)
        positional = _replace_positional(corpus, 2, tuple(matrix))
    elif case == "catalog-id-pin-str":
        keyword["out_of_band_expected_catalog_id_bytes"] = CATALOG_ID
    else:
        keyword["out_of_band_expected_catalog_payload_sha256_digest"] = bytearray(
            bytes.fromhex(CATALOG_SHA256)
        )
    _assert_error(
        expected_code,
        lambda: _verify(corpus, positional=positional, keyword_only=keyword),
    )


@pytest.mark.parametrize(
    "case",
    ("missing", "extra", "swap", "duplicate", "body-array-not-list", "envelope-missing", "envelope-extra"),
)
def test_04_source_missing_extra_swap_duplicate_and_exact_json_shapes_reject(
    corpus: _Corpus, case: str
) -> None:
    raws = list(corpus.positional[1])
    expected = mechanics.FAIL_TYPE_V1 if case in {"missing", "extra"} else mechanics.FAIL_BINDING_V1
    if case == "missing":
        raws.pop()
    elif case == "extra":
        raws.append(raws[-1])
    elif case == "swap":
        raws[0], raws[1] = raws[1], raws[0]
    elif case == "duplicate":
        raws[1] = raws[0]
    elif case == "body-array-not-list":
        raws[0] = _rewrite_source(
            raws[0],
            lambda body: body.__setitem__("predecessor_ordinals_in_exact_order", {}),
            rehash=True,
        )
    else:
        envelope = _decoded(raws[0])
        if case == "envelope-missing":
            envelope.pop("schema_version")
        else:
            envelope["extra"] = False
        raws[0] = _canonical(envelope)
        expected = mechanics.FAIL_SCHEMA_V1
    if case == "body-array-not-list":
        expected = mechanics.FAIL_SCHEMA_V1
    _assert_error(
        expected,
        lambda: _verify(
            corpus,
            positional=_replace_positional(corpus, 1, tuple(raws)),
        ),
    )


def test_05_opaque_id_sha_only_composite_cannot_replace_full_raw_leaves(
    corpus: _Corpus,
) -> None:
    # The source-4 manifest still carries IDs, SHAs, counts, and a root.  With
    # the six raw payloads removed it is exactly the forbidden opaque form.
    _assert_error(
        mechanics.FAIL_COMPOSITE_V1,
        lambda: _verify(
            corpus,
            positional=_replace_leaf_row(corpus, 4, ()),
        ),
    )

    source = _decoded(corpus.positional[1][4])
    body = source["body"]
    assert type(body) is dict
    body["composite_leaf_manifest_or_none"] = {
        "opaque_bundle_id": "opaque-id-only",
        "opaque_bundle_sha256": "sha256:" + "0" * 64,
    }
    _assert_error(
        mechanics.FAIL_SCHEMA_V1,
        lambda: _verify(
            corpus,
            positional=_replace_source_raw(corpus, 4, _canonical(source)),
        ),
    )


@pytest.mark.parametrize("case", ("missing", "reordered", "mutated", "runtime-copy-drift"))
def test_06_complete_raw_leaf_sequences_reject_missing_reorder_and_mutation(
    corpus: _Corpus, case: str
) -> None:
    matrix = list(corpus.positional[2])
    expected = mechanics.FAIL_COMPOSITE_V1
    if case == "missing":
        matrix[22] = matrix[22][:-1]
    elif case == "reordered":
        row = list(matrix[4])
        row[0], row[1] = row[1], row[0]
        matrix[4] = tuple(row)
    elif case == "mutated":
        row = list(matrix[7])
        row[0] = _rewrite_leaf(
            row[0],
            7,
            0,
            lambda body: body.__setitem__("payload_label", "TEST_ONLY_MUTATED"),
            rehash=True,
        )
        matrix[7] = tuple(row)
    else:
        row = list(matrix[16])
        row[0] = bytes(row[0] + b" ")
        matrix[16] = tuple(row)
    _assert_error(
        expected,
        lambda: _verify(
            corpus,
            positional=_replace_positional(corpus, 2, tuple(matrix)),
        ),
    )


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("source-id", mechanics.FAIL_CONTENT_ID_V1),
        ("leaf-id", mechanics.FAIL_CONTENT_ID_V1),
        ("manifest-sha", mechanics.FAIL_COMPOSITE_V1),
        ("manifest-count", mechanics.FAIL_COMPOSITE_V1),
        ("manifest-root", mechanics.FAIL_COMPOSITE_V1),
    ),
)
def test_07_content_ids_leaf_sha_byte_count_and_root_are_recomputed(
    corpus: _Corpus, case: str, expected: str
) -> None:
    if case == "source-id":
        envelope = _decoded(corpus.positional[1][0])
        envelope["native_source_id"] = NATIVE_ID_PREFIX + "0" * 64
        positional = _replace_source_raw(corpus, 0, _canonical(envelope))
    elif case == "leaf-id":
        matrix = list(corpus.positional[2])
        row = list(matrix[4])
        envelope = _decoded(row[0])
        envelope["raw_leaf_id"] = LEAF_ID_PREFIX + "0" * 64
        row[0] = _canonical(envelope)
        matrix[4] = tuple(row)
        positional = _replace_positional(corpus, 2, tuple(matrix))
    else:
        envelope = _decoded(corpus.positional[1][4])
        body = envelope["body"]
        assert type(body) is dict
        manifest = body["composite_leaf_manifest_or_none"]
        assert type(manifest) is dict
        if case == "manifest-sha":
            values = manifest["leaf_payload_sha256s_in_exact_order"]
            assert type(values) is list
            values[0] = "sha256:" + "0" * 64
        elif case == "manifest-count":
            values = manifest["leaf_payload_byte_counts_in_exact_order"]
            assert type(values) is list and type(values[0]) is int
            values[0] += 1
        else:
            manifest["leaf_root_sha256"] = "sha256:" + "0" * 64
        positional = _replace_source_raw(corpus, 4, _canonical(envelope))
    _assert_error(expected, lambda: _verify(corpus, positional=positional))


@pytest.mark.parametrize("case", ("ordinal", "id", "sha", "cross-corpus"))
def test_08_predecessor_splices_reject_before_any_claim(
    corpus: _Corpus, case: str
) -> None:
    envelope = _decoded(corpus.positional[1][23])
    body = envelope["body"]
    assert type(body) is dict
    if case == "ordinal":
        values = body["predecessor_ordinals_in_exact_order"]
        assert type(values) is list
        values[0], values[1] = values[1], values[0]
    elif case == "id":
        values = body["predecessor_source_ids_in_exact_order"]
        assert type(values) is list
        values[4] = corpus.source_ids[0]
    elif case == "sha":
        values = body["predecessor_source_payload_sha256s_in_exact_order"]
        assert type(values) is list
        values[7] = "sha256:" + "0" * 64
    else:
        values = body["predecessor_source_ids_in_exact_order"]
        assert type(values) is list
        values[4] = NATIVE_ID_PREFIX + hashlib.sha256(
            b"TEST_ONLY_HYPOTHETICAL_SECOND_CORPUS_SOURCE_4"
        ).hexdigest()
    _assert_error(
        mechanics.FAIL_BINDING_V1,
        lambda: _verify(
            corpus,
            positional=_replace_source_raw(corpus, 23, _canonical(envelope)),
        ),
    )


@pytest.mark.parametrize("ordinal", (4, 5))
def test_09_presented_active_zero_revocation_profile_drift_rejects(
    corpus: _Corpus, ordinal: int
) -> None:
    mutated = _rewrite_source(
        corpus.positional[1][ordinal],
        lambda body: body.__setitem__(
            "presented_state_profile",
            "PRESENTED_REVOKED_OR_TERMINAL_PROFILE_FORBIDDEN",
        ),
        rehash=True,
    )
    _assert_error(
        mechanics.FAIL_BINDING_V1,
        lambda: _verify(
            corpus, positional=_replace_source_raw(corpus, ordinal, mutated)
        ),
    )


@pytest.mark.parametrize("case", ("true", "int-zero", "missing", "extra", "leaf-true"))
def test_10_every_nonclaim_is_exact_bool_false_on_sources_and_raw_leaves(
    corpus: _Corpus, case: str
) -> None:
    if case == "leaf-true":
        matrix = list(corpus.positional[2])
        row = list(matrix[4])
        row[0] = _rewrite_leaf(
            row[0],
            4,
            0,
            lambda body: body["nonclaims"].__setitem__(  # type: ignore[union-attr]
                "owner_non_revocation_verified", True
            ),
        )
        matrix[4] = tuple(row)
        positional = _replace_positional(corpus, 2, tuple(matrix))
    else:
        def mutate(body: dict[str, object]) -> None:
            values = body["nonclaims"]
            assert type(values) is dict
            if case == "true":
                values["external_currentness_verified"] = True
            elif case == "int-zero":
                values["authority_verified"] = 0
            elif case == "missing":
                values.pop("c1_satisfied")
            else:
                values["invented_claim"] = False

        positional = _replace_source_raw(
            corpus,
            0,
            _rewrite_source(corpus.positional[1][0], mutate, rehash=True),
        )
    _assert_error(
        mechanics.FAIL_CLAIM_BOUNDARY_V1,
        lambda: _verify(corpus, positional=positional),
    )


def test_11_private_constructor_forgery_copy_pickle_and_mutation_reject(
    corpus: _Corpus,
) -> None:
    result = _verify(corpus)
    result_type = mechanics.VerifiedPhase2BActualUnsealed960R01PresentedMechanicsV1
    with pytest.raises(TypeError, match="privately issued"):
        result_type()
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(result)
    with pytest.raises(TypeError, match="cannot be deep-copied"):
        copy.deepcopy(result)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(result)
    forged = object.__new__(result_type)
    _assert_error(
        mechanics.FAIL_RESULT_V1,
        lambda: mechanics.require_verified_phase2b_actual_unsealed_960_r01_presented_mechanics_v1(
            forged
        ),
    )
    object.__setattr__(result, "r01_satisfied", True)
    _assert_error(
        mechanics.FAIL_RESULT_V1,
        lambda: mechanics.require_verified_phase2b_actual_unsealed_960_r01_presented_mechanics_v1(
            result
        ),
    )

    nested_mutated = _verify(corpus)
    nested_ids = list(nested_mutated.native_source_ids_in_exact_order)
    nested_ids[0] = _EqAll()  # type: ignore[assignment]
    object.__setattr__(
        nested_mutated,
        "native_source_ids_in_exact_order",
        tuple(nested_ids),
    )
    _assert_error(
        mechanics.FAIL_RESULT_V1,
        lambda: mechanics.require_verified_phase2b_actual_unsealed_960_r01_presented_mechanics_v1(
            nested_mutated
        ),
    )


def test_12_same_abi_real_entry_is_exact_not_implemented(corpus: _Corpus) -> None:
    assert mechanics.REAL_VERIFICATION_STATUS_V1 == "NOT_IMPLEMENTED"
    with pytest.raises(
        mechanics.Phase2BActualUnsealed960R01PresentedMechanicsErrorV1,
        match=(
            r"^FAIL_R01_REAL_CURRENT_CUT_ACTIVE_NONREVOKED_V1_NOT_IMPLEMENTED: "
            r"real R01 requires external currentness, authority, custody, "
            r"durability, and non-revocation evidence$"
        ),
    ):
        mechanics.verify_real_phase2b_actual_unsealed_960_r01_current_cut_active_nonrevoked_v1(
            *corpus.positional, **corpus.keyword_only
        )


def test_13_config_is_exact_contract_bytes_without_lf() -> None:
    raw = CONFIG.read_bytes()
    assert raw == mechanics.CONTRACT_PAYLOAD_V1
    assert not raw.endswith(b"\n")
    assert hashlib.sha256(raw).hexdigest() == mechanics.CONTRACT_PAYLOAD_SHA256_V1
    assert decode_and_require_canonical_rfc8785_v1(raw) == {
        "schema_version": mechanics.CONTRACT_SCHEMA_VERSION_V1,
        "body": mechanics.CONTRACT_BODY_V1,
        "contract_id": mechanics.CONTRACT_ID_V1,
    }


def test_14_post_import_rebind_cannot_change_definition_time_boundary(
    corpus: _Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = mechanics.verify_phase2b_actual_unsealed_960_r01_real_current_cut_active_nonrevoked_presented_mechanics_v1
    require_result = mechanics.require_verified_phase2b_actual_unsealed_960_r01_presented_mechanics_v1

    def bomb(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("post-import rebound dependency was consulted")

    monkeypatch.setattr(rfc8785_module, "canonicalize_rfc8785_v1", bomb)
    monkeypatch.setattr(rfc8785_module, "decode_and_require_canonical_rfc8785_v1", bomb)
    monkeypatch.setattr(rfc8785_module, "_serialize", bomb)
    monkeypatch.setattr(rfc8785_module, "_decode_json_bytes", bomb)
    monkeypatch.setattr(mechanics, "_CAPTURED_CANONICALIZE_V1", bomb)
    monkeypatch.setattr(mechanics, "_CAPTURED_DECODE_V1", bomb)
    monkeypatch.setattr(mechanics, "_catalog", None)
    monkeypatch.setattr(mechanics, "_rfc8785", None)
    monkeypatch.setattr(mechanics, "hashlib", None)
    for name in (
        "SCHEMA_VERSION_V1",
        "STATUS_V1",
        "CLAIM_LEVEL_V1",
        "CATALOG_R01_GATE_ORDINAL_V1",
        "CATALOG_R01_GATE_ID_V1",
        "CATALOG_R01_DISPATCH_ID_V1",
        "NATIVE_SOURCE_ENVELOPE_SCHEMA_VERSION_V1",
        "NATIVE_SOURCE_BODY_SCHEMA_VERSION_V1",
        "RAW_COMPOSITE_LEAF_ENVELOPE_SCHEMA_VERSION_V1",
        "RAW_COMPOSITE_LEAF_BODY_SCHEMA_VERSION_V1",
        "COMPOSITE_MANIFEST_SCHEMA_VERSION_V1",
        "NATIVE_SOURCE_STATUS_V1",
        "NATIVE_SOURCE_CLAIM_LEVEL_V1",
        "RAW_LEAF_STATUS_V1",
        "RAW_LEAF_CLAIM_LEVEL_V1",
        "NATIVE_SOURCE_ID_PREFIX_V1",
        "RAW_COMPOSITE_LEAF_ID_PREFIX_V1",
        "NATIVE_SOURCE_ID_DOMAIN_PREFIX_V1",
        "RAW_COMPOSITE_LEAF_ID_DOMAIN_PREFIX_V1",
        "COMPOSITE_LEAF_ROOT_DOMAIN_PREFIX_V1",
        "SOURCE_GRAPH_ROOT_DOMAIN_V1",
    ):
        monkeypatch.setattr(mechanics, name, "POISONED_AFTER_IMPORT")
    monkeypatch.setattr(mechanics, "MAXIMUM_JSON_SAFE_INTEGER_V1", 0)
    monkeypatch.setattr(
        mechanics,
        "verify_phase2b_actual_unsealed_960_r01_real_current_cut_active_nonrevoked_presented_mechanics_v1",
        bomb,
    )
    monkeypatch.setattr(
        mechanics,
        "require_verified_phase2b_actual_unsealed_960_r01_presented_mechanics_v1",
        bomb,
    )
    result = verifier(*corpus.positional, **corpus.keyword_only)
    assert require_result(result) is result
    assert result.source_graph_root_sha256 == corpus.graph_root


@pytest.mark.parametrize(
    ("poison", "expected_error"),
    (
        ("payload", "R01 downstream catalog payload SHA differs"),
        ("public-spec", "R01 downstream mutable catalog surface differs"),
    ),
)
def test_15_preimport_catalog_poisoning_rejects_controlled(
    poison: str, expected_error: str
) -> None:
    catalog_module = (
        "hegel_machine.phase2b_actual_unsealed_960_"
        "authoritative_evidence_schema_catalog_v1"
    )
    target_module = (
        "hegel_machine.phase2b_actual_unsealed_960_verify_r01_real_current_cut_"
        "active_nonrevoked_v1"
    )
    mutation = (
        "catalog.CATALOG_PAYLOAD_V1 = catalog.CATALOG_PAYLOAD_V1 + b' '"
        if poison == "payload"
        else (
            "object.__setattr__(catalog.GATE_SOURCE_SPECS_IN_EXACT_ORDER_V1[0], "
            "'verifier_dispatch_id', 'POISONED_PREIMPORT_DISPATCH')"
        )
    )
    child = (
        "import importlib\n"
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        f"catalog = importlib.import_module({catalog_module!r})\n"
        f"assert {target_module!r} not in sys.modules\n"
        f"{mutation}\n"
        "try:\n"
        f"    importlib.import_module({target_module!r})\n"
        "except RuntimeError as exc:\n"
        f"    if str(exc) != {expected_error!r}:\n"
        "        raise AssertionError(repr(str(exc))) from exc\n"
        "    print(str(exc))\n"
        "else:\n"
        "    raise AssertionError('poisoned catalog import unexpectedly succeeded')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", child],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == expected_error + "\n"


def test_16_source_ast_and_fresh_import_are_offline_producer_free() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    forbidden_modules = {
        "os",
        "pathlib",
        "socket",
        "subprocess",
        "http",
        "urllib",
        "requests",
        "docker",
        "time",
        "datetime",
        "random",
        "secrets",
        "tempfile",
        "sqlite3",
        "asyncio",
        "multiprocessing",
    }
    imported_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.append(node.module)
            imported_names.extend(alias.name for alias in node.names)
    assert not any(name.split(".")[0] in forbidden_modules for name in imported_names)
    producer_fragments = (
        "owner_status",
        "logical_time",
        "runtime_closure",
        "cas_store",
        "pure_replay",
        "evidence_contract",
    )
    assert not any(
        fragment in name
        for name in imported_names
        for fragment in producer_fragments
    )
    forbidden_calls = {
        "eval",
        "system",
        "popen",
        "fork",
        "sign",
        "read_bytes",
        "write_bytes",
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called.update(
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )
    assert called.isdisjoint(forbidden_calls)

    # Package0-style import-time capture may read and compile the already
    # installed local RFC8785 dependency.  The actual verifier runtime below
    # the explicit private-capture boundary must not perform those operations.
    source_lines = SOURCE.read_text(encoding="utf-8").splitlines()
    runtime_start = next(
        index
        for index, line in enumerate(source_lines, start=1)
        if "PRIVATE_RUNTIME_CAPTURE_START" in line
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        else:
            continue
        if call_name in {"open", "compile", "exec"}:
            assert node.lineno < runtime_start
    assert mechanics.CONTRACT_BODY_V1[
        "decoded_mapping_path_key_callback_socket_clock_nonce_inputs_allowed"
    ] is False

    target_module = (
        "hegel_machine.phase2b_actual_unsealed_960_verify_r01_real_current_cut_"
        "active_nonrevoked_v1"
    )
    child = (
        "import importlib\n"
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        f"m = importlib.import_module({target_module!r})\n"
        "print(m.CONTRACT_PAYLOAD_SHA256_V1)\n"
        "print(m.REAL_VERIFICATION_STATUS_V1)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", child],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={"PATH": os.environ.get("PATH", "")},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (
        mechanics.CONTRACT_PAYLOAD_SHA256_V1 + "\nNOT_IMPLEMENTED\n"
    )


def test_17_sixteen_mib_plus_one_byte_boundary_rejects_before_decode(
    corpus: _Corpus,
) -> None:
    oversized = b"x" * (
        mechanics.MAXIMUM_EXACT_BYTES_PARAMETER_OR_TUPLE_ITEM_BYTES_V1 + 1
    )
    _assert_error(
        mechanics.FAIL_TYPE_V1,
        lambda: _verify(
            corpus,
            positional=_replace_positional(corpus, 0, oversized),
        ),
    )
    raws = list(corpus.positional[1])
    raws[0] = oversized
    _assert_error(
        mechanics.FAIL_TYPE_V1,
        lambda: _verify(
            corpus,
            positional=_replace_positional(corpus, 1, tuple(raws)),
        ),
    )
