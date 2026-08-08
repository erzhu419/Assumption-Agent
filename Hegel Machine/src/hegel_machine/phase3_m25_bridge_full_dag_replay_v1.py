"""Independent, fail-closed M2.5 bridge-candidate DAG replay.

The replay package contains every candidate-addressed public/static/typed
preimage.  Six hidden split trees are deliberately represented only by their
sealed RFC6962 roots and frozen row counts.  Purpose 1 may inspect an unsigned
package inside its custody boundary; purposes 2 and 3 must additionally replay
an Ed25519 purpose-1 signature over the exact bridge statement root.

This module neither signs nor creates authoritative material.  ``authority``
is frozen into the package and authoritative packages require an explicit
integration-only opt in.  A successful replay therefore means root/DAG and
signature binding, never reconstruction of undisclosed split membership.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from types import MappingProxyType
from typing import Callable, Final, Mapping, NoReturn, Sequence

from .phase3_m25_wire_v1 import (
    FORMAL_SCHEMA_REGISTRY,
    bridge_attestation_signature_preimage_v1,
    candidate_content_root,
    decode_formal_object,
    encode_formal_object,
)
from .strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
    rfc6962_root,
)


PACKAGE_TAG: Final = 0x3501
PACKAGE_SCHEMA_ID: Final = b"hegel-m25-bridge-full-dag-replay-package/1"
PACKAGE_HASH_DOMAIN: Final = "HEGEL/M25/BRIDGE_FULL_DAG_REPLAY_PACKAGE/V1"
SEALED_SPLIT_SCHEMA_ID: Final = b"hegel-sealed-split-commitment/1"
CANONICAL_INPUT_DOMAIN: Final = "HEGEL/CANONICAL_INPUT/V1"
ACTOR_REPLAY_RECEIPT_SCHEMA: Final = (
    "hegel-phase3-m25-bridge-dag-actor-replay-receipt/1"
)
ACTOR_REPLAY_IMPLEMENTATIONS: Final = (
    "python-full-dag-replay-v1",
    "rust-full-dag-replay-v1",
)

OP_CONTENT: Final = 1
OP_RFC6962: Final = 2
OP_SEALED_SPLIT: Final = 3

FAIL_PACKAGE_SCHEMA: Final = "FAIL_M25_BRIDGE_REPLAY_PACKAGE_SCHEMA"
FAIL_PACKAGE_AUTHORITY: Final = "FAIL_M25_BRIDGE_REPLAY_AUTHORITY_GUARD"
FAIL_PURPOSE: Final = "FAIL_M25_BRIDGE_REPLAY_PURPOSE"
FAIL_NODE_SET: Final = "FAIL_M25_BRIDGE_REPLAY_NODE_SET"
FAIL_NODE_SCHEMA: Final = "FAIL_M25_BRIDGE_REPLAY_NODE_SCHEMA"
FAIL_NODE_PREIMAGE: Final = "FAIL_M25_BRIDGE_REPLAY_NODE_PREIMAGE"
FAIL_NODE_COUNT: Final = "FAIL_M25_BRIDGE_REPLAY_NODE_COUNT"
FAIL_ROOT_BINDING: Final = "FAIL_M25_BRIDGE_REPLAY_ROOT_BINDING"
FAIL_CANDIDATE: Final = "FAIL_M25_BRIDGE_REPLAY_CANDIDATE"
FAIL_BRIDGE: Final = "FAIL_M25_BRIDGE_REPLAY_BRIDGE"
FAIL_ROLE_BINDING: Final = "FAIL_M25_BRIDGE_REPLAY_CROSS_ROLE"
FAIL_TYPED_BINDING: Final = "FAIL_M25_BRIDGE_REPLAY_TYPED_BINDING"
FAIL_SPLIT_BINDING: Final = "FAIL_M25_BRIDGE_REPLAY_SEALED_SPLIT_BINDING"
FAIL_TRUST_BINDING: Final = "FAIL_M25_BRIDGE_REPLAY_PURPOSE1_TRUST_BINDING"
FAIL_SIGNATURE_PHASE: Final = "FAIL_M25_BRIDGE_REPLAY_SIGNATURE_PHASE"
FAIL_SIGNATURE: Final = "FAIL_M25_BRIDGE_REPLAY_PURPOSE1_SIGNATURE"
FAIL_ACTOR_RECEIPT: Final = "FAIL_M25_BRIDGE_REPLAY_ACTOR_RECEIPT"

OPENSSL_EXECUTABLE: Final = Path("/usr/bin/openssl")
OPENSSL_EXECUTABLE_SHA256: Final = (
    "a55e3085b6a1df8887722f6cee7fc32c861d11d5fb584a63837d32d29602c65b"
)
_MAX_OPENSSL_EXECUTABLE_BYTES: Final = 16 * 1024 * 1024

Ed25519VerifierV1 = Callable[[bytes, bytes, bytes], None]


class BridgeDagReplayError(RuntimeError):
    """Stable fail-closed replay error."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise BridgeDagReplayError(code, detail)


@dataclass(frozen=True, slots=True)
class ReplayRoleSpecV1:
    role_id: int
    candidate_index: int
    field_name: str
    operation_id: int
    schema_name: str | None
    exact_count: int

    @property
    def tag(self) -> int:
        return 0 if self.schema_name is None else FORMAL_SCHEMA_REGISTRY[self.schema_name].tag

    @property
    def schema_id(self) -> bytes:
        return (
            SEALED_SPLIT_SCHEMA_ID
            if self.schema_name is None
            else FORMAL_SCHEMA_REGISTRY[self.schema_name].schema_id
        )

    @property
    def domain(self) -> bytes | None:
        if self.schema_name is None:
            return None
        value = FORMAL_SCHEMA_REGISTRY[self.schema_name].hash_domain
        return None if value is None else value.encode("ascii")

    @property
    def field_count(self) -> int:
        return 0 if self.schema_name is None else len(FORMAL_SCHEMA_REGISTRY[self.schema_name].fields)


_ROLE_ROWS = (
    (1, 4, "child_dsl_spec_root", OP_CONTENT, "DslSpecV1", 1),
    (2, 5, "child_freeze_root", OP_CONTENT, "FreezeSpecV1", 1),
    (3, 6, "approval_manifest_root", OP_CONTENT, "NormativeApprovalManifestV1", 1),
    (4, 7, "shrink_transition_root", OP_CONTENT, "DslShrinkTransitionFormalV1", 1),
    (5, 8, "operator_semantics_root", OP_RFC6962, "OperatorSemanticsEntryV1", 28),
    (6, 9, "identifier_registry_root", OP_RFC6962, "IdentifierRegistryEntryV1", 55),
    (7, 10, "canonical_ast_schema_root", OP_CONTENT, "CanonicalAstProfileSpecV1", 1),
    (8, 11, "canonical_cbor_profile_root", OP_CONTENT, "CanonicalCborProfileSpecV1", 1),
    (9, 12, "diagnostic_formal_bridge_root", OP_RFC6962, "DiagnosticFormalBridgeRecordV1", 12),
    (10, 13, "outside_target_binding_manifest_root", OP_CONTENT, "DslRoleBindingManifestV1", 1),
    (11, 14, "null_control_binding_manifest_root", OP_CONTENT, "DslRoleBindingManifestV1", 1),
    (12, 15, "split_binding_manifest_root", OP_CONTENT, "SplitBindingManifestV1", 1),
    (13, 16, "custodian_binding_manifest_root", OP_CONTENT, "CustodianBindingManifestV1", 1),
    (14, 17, "seed_continuity_manifest_root", OP_CONTENT, "SeedContinuityManifestV1", 1),
    (15, 18, "custodian_attestation_bundle_root", OP_CONTENT, "AttestationBundleV1", 1),
    (16, 19, "parent_absence_attestation_root", OP_CONTENT, "ParentManifestAbsenceAttestationV2", 1),
    (17, 20, "hidden_access_ledger_genesis_root", OP_CONTENT, "HiddenAccessLedgerRecordV1", 1),
    (18, 21, "hidden_access_ledger_head_root", OP_CONTENT, "HiddenAccessLedgerRecordV1", 1),
    (19, 22, "opaque_id_registry_snapshot_root", OP_CONTENT, "OpaqueIdRegistrySnapshotV1", 1),
    (20, 23, "actor_trust_genesis_root", OP_CONTENT, "ActorTrustGenesisV1", 1),
    (21, 24, "outside_target_universe_root", OP_RFC6962, "BoundedUniverseRowV1", 480),
    (22, 25, "outside_target_truth_root", OP_RFC6962, "TargetTruthRowV1", 480),
    (23, 26, "null_control_universe_root", OP_RFC6962, "BoundedUniverseRowV1", 85),
    (24, 27, "null_control_truth_root", OP_RFC6962, "TargetTruthRowV1", 85),
    (25, 28, "outside_discovery_split_root", OP_SEALED_SPLIT, None, 192),
    (26, 29, "outside_validation_split_root", OP_SEALED_SPLIT, None, 96),
    (27, 30, "outside_sealed_split_root", OP_SEALED_SPLIT, None, 192),
    (28, 31, "null_discovery_split_root", OP_SEALED_SPLIT, None, 39),
    (29, 32, "null_validation_split_root", OP_SEALED_SPLIT, None, 20),
    (30, 33, "null_sealed_split_root", OP_SEALED_SPLIT, None, 26),
    (31, 38, "python_implementation_binding_root", OP_CONTENT, "ImplementationBindingV1", 1),
    (32, 39, "rust_implementation_binding_root", OP_CONTENT, "ImplementationBindingV1", 1),
    (33, 40, "traversal_contract_root", OP_CONTENT, "TraversalContractV1", 1),
    (34, 41, "bucket_accounting_contract_root", OP_CONTENT, "BucketAccountingContractV1", 1),
    (35, 42, "program_archive_contract_root", OP_CONTENT, "ProgramArchiveContractV1", 1),
    (36, 43, "output_archive_contract_root", OP_CONTENT, "OutputArchiveContractV1", 1),
    (37, 44, "state_machine_contract_root", OP_CONTENT, "StateMachineContractV1", 1),
)

ROLE_SPECS: Final = tuple(ReplayRoleSpecV1(*row) for row in _ROLE_ROWS)
ROLE_SPEC_BY_ID: Final = MappingProxyType({row.role_id: row for row in ROLE_SPECS})


@dataclass(frozen=True, slots=True)
class ReplayNodeV1:
    role_id: int
    preimages: tuple[bytes, ...] = ()
    sealed_root: bytes | None = None


@dataclass(frozen=True, slots=True)
class BridgeDagReplayResultV1:
    package_digest: bytes
    candidate_root: bytes
    bridge_statement_root: bytes
    purpose_id: int
    purpose1_signature_verified: bool
    eligible_to_sign_bridge_statement: bool
    authoritative: bool
    split_membership_recomputed: bool = False
    split_claim: str = "SEALED_ROOT_COUNT_AND_PURPOSE1_BINDING_ONLY"


def _canonical_json_line(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def build_bridge_actor_replay_receipt_v1(
    result: BridgeDagReplayResultV1,
    *,
    implementation: str,
) -> bytes:
    """Build the one-line public receipt emitted beside an actor signature."""

    if implementation not in ACTOR_REPLAY_IMPLEMENTATIONS:
        _fail(FAIL_ACTOR_RECEIPT, "actor replay implementation is not registered")
    body: dict[str, object] = {
        "authoritative": result.authoritative,
        "bridge_statement_root_hex": result.bridge_statement_root.hex(),
        "candidate_root_hex": result.candidate_root.hex(),
        "eligible_to_sign_bridge_statement": result.eligible_to_sign_bridge_statement,
        "implementation": implementation,
        "package_digest_hex": result.package_digest.hex(),
        "purpose": result.purpose_id,
        "purpose1_signature_verified": result.purpose1_signature_verified,
        "schema": ACTOR_REPLAY_RECEIPT_SCHEMA,
        "signing_key_epoch": 0,
        "split_claim": result.split_claim,
        "split_membership_recomputed": result.split_membership_recomputed,
        "status": "PASS",
    }
    body["receipt_sha256"] = hashlib.sha256(_canonical_json_line(body)).hexdigest()
    return _canonical_json_line(body)


def _strict_json_object(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes or not payload or len(payload) > 16 * 1024:
        _fail(FAIL_ACTOR_RECEIPT, "actor receipt byte length is invalid")
    try:
        text = payload.decode("ascii", "strict")

        def reject_float(_value: str) -> NoReturn:
            _fail(FAIL_ACTOR_RECEIPT, "actor receipt contains a non-integer number")

        def reject_constant(_value: str) -> NoReturn:
            _fail(FAIL_ACTOR_RECEIPT, "actor receipt contains a non-JSON constant")

        def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            value: dict[str, object] = {}
            for key, item in pairs:
                if key in value:
                    _fail(FAIL_ACTOR_RECEIPT, "actor receipt contains a duplicate key")
                value[key] = item
            return value

        value = json.loads(
            text,
            parse_float=reject_float,
            parse_constant=reject_constant,
            object_pairs_hook=exact_object,
        )
    except BridgeDagReplayError:
        raise
    except Exception as exc:
        _fail(FAIL_ACTOR_RECEIPT, f"actor receipt is not strict ASCII JSON: {exc}")
    if type(value) is not dict or _canonical_json_line(value) != payload:
        _fail(FAIL_ACTOR_RECEIPT, "actor receipt is not one canonical JSON line")
    return value


def validate_bridge_actor_replay_receipt_v1(
    payload: bytes,
    *,
    expected_result: BridgeDagReplayResultV1 | None = None,
    expected_implementation: str | None = None,
    require_authoritative: bool = False,
) -> Mapping[str, object]:
    """Strictly validate and optionally bind one actor replay receipt."""

    value = _strict_json_object(payload)
    expected_fields = {
        "authoritative",
        "bridge_statement_root_hex",
        "candidate_root_hex",
        "eligible_to_sign_bridge_statement",
        "implementation",
        "package_digest_hex",
        "purpose",
        "purpose1_signature_verified",
        "receipt_sha256",
        "schema",
        "signing_key_epoch",
        "split_claim",
        "split_membership_recomputed",
        "status",
    }
    if set(value) != expected_fields:
        _fail(FAIL_ACTOR_RECEIPT, "actor receipt field set differs")
    if (
        type(value["authoritative"]) is not bool
        or type(value["eligible_to_sign_bridge_statement"]) is not bool
        or type(value["purpose1_signature_verified"]) is not bool
        or type(value["split_membership_recomputed"]) is not bool
        or type(value["purpose"]) is not int
        or type(value["signing_key_epoch"]) is not int
        or type(value["implementation"]) is not str
        or value["implementation"] not in ACTOR_REPLAY_IMPLEMENTATIONS
        or value["purpose"] not in (1, 2, 3)
        or value["signing_key_epoch"] != 0
        or value["eligible_to_sign_bridge_statement"] is not True
        or value["purpose1_signature_verified"] is not (value["purpose"] != 1)
        or value["split_membership_recomputed"] is not False
        or value["split_claim"] != "SEALED_ROOT_COUNT_AND_PURPOSE1_BINDING_ONLY"
        or value["schema"] != ACTOR_REPLAY_RECEIPT_SCHEMA
        or value["status"] != "PASS"
    ):
        _fail(FAIL_ACTOR_RECEIPT, "actor receipt structural value differs")
    for field in (
        "bridge_statement_root_hex",
        "candidate_root_hex",
        "package_digest_hex",
        "receipt_sha256",
    ):
        if type(value[field]) is not str or re.fullmatch(r"[0-9a-f]{64}", value[field]) is None:
            _fail(FAIL_ACTOR_RECEIPT, f"actor receipt field {field} is not lowercase hex32")
    supplied_digest = value["receipt_sha256"]
    body = dict(value)
    del body["receipt_sha256"]
    if supplied_digest != hashlib.sha256(_canonical_json_line(body)).hexdigest():
        _fail(FAIL_ACTOR_RECEIPT, "actor receipt self-hash differs")
    if require_authoritative and value["authoritative"] is not True:
        _fail(FAIL_ACTOR_RECEIPT, "formal actor receipt is not authoritative")
    if expected_implementation is not None and value["implementation"] != expected_implementation:
        _fail(FAIL_ACTOR_RECEIPT, "actor receipt implementation differs")
    if expected_result is not None:
        expected_values = {
            "authoritative": expected_result.authoritative,
            "bridge_statement_root_hex": expected_result.bridge_statement_root.hex(),
            "candidate_root_hex": expected_result.candidate_root.hex(),
            "eligible_to_sign_bridge_statement": expected_result.eligible_to_sign_bridge_statement,
            "package_digest_hex": expected_result.package_digest.hex(),
            "purpose": expected_result.purpose_id,
            "purpose1_signature_verified": expected_result.purpose1_signature_verified,
            "split_claim": expected_result.split_claim,
            "split_membership_recomputed": expected_result.split_membership_recomputed,
        }
        if any(value[field] != item for field, item in expected_values.items()):
            _fail(FAIL_ACTOR_RECEIPT, "actor receipt differs from replay result")
    return MappingProxyType(dict(value))


def _node_wire(node: ReplayNodeV1) -> tuple[object, ...]:
    spec = ROLE_SPEC_BY_ID.get(node.role_id)
    if spec is None:
        _fail(FAIL_NODE_SET, f"unknown replay role {node.role_id}")
    if spec.operation_id == OP_SEALED_SPLIT:
        if node.preimages or type(node.sealed_root) is not bytes or len(node.sealed_root) != 32:
            _fail(FAIL_NODE_PREIMAGE, f"sealed role {node.role_id} has invalid disclosure")
    elif node.sealed_root is not None:
        _fail(FAIL_NODE_PREIMAGE, f"public role {node.role_id} supplies a sealed root")
    return (
        node.role_id,
        spec.operation_id,
        spec.tag,
        spec.schema_id,
        spec.domain,
        tuple(node.preimages),
        spec.exact_count,
        node.sealed_root,
    )


def build_bridge_dag_replay_package_v1(
    *,
    purpose_id: int,
    candidate_fields: Mapping[str, object],
    bridge_statement_fields: Mapping[str, object],
    nodes: Sequence[ReplayNodeV1],
    purpose1_actor_key_manifest_fields: Mapping[str, object],
    purpose1_bridge_signature: bytes | None,
    authority: bool = False,
) -> bytes:
    """Build exact package bytes; validation remains a separate mandatory step."""

    candidate = encode_formal_object("M3ExecutionCandidateV1", candidate_fields)
    bridge = encode_formal_object("BridgeReplayStatementV1", bridge_statement_fields)
    key_manifest = encode_formal_object("ActorKeyManifestV1", purpose1_actor_key_manifest_fields)
    wire = (
        1,
        PACKAGE_TAG,
        PACKAGE_SCHEMA_ID,
        authority,
        purpose_id,
        candidate,
        bridge,
        tuple(_node_wire(node) for node in nodes),
        key_manifest,
        purpose1_bridge_signature,
        candidate_fields["created_at_unix_seconds"],
        candidate_fields["repository_commit_id"],
    )
    return canonical_cbor_encode(wire)


def _require_bytes(value: object, length: int, code: str, label: str) -> bytes:
    if type(value) is not bytes or len(value) != length:
        _fail(code, f"{label} must be exactly {length} bytes")
    return value


def _decode_node(role: ReplayRoleSpecV1, raw: object) -> tuple[tuple[bytes, ...], bytes | None]:
    if not isinstance(raw, tuple) or len(raw) != 8:
        _fail(FAIL_NODE_SCHEMA, f"role {role.role_id} node is not the eight-field array")
    role_id, operation, tag, schema_id, domain, preimages_raw, count, sealed_root = raw
    expected = (role.role_id, role.operation_id, role.tag, role.schema_id, role.domain)
    if (role_id, operation, tag, schema_id, domain) != expected:
        _fail(FAIL_NODE_SCHEMA, f"role {role.role_id} operation/schema/domain differs")
    if count != role.exact_count:
        _fail(FAIL_NODE_COUNT, f"role {role.role_id} count differs from {role.exact_count}")
    if not isinstance(preimages_raw, tuple) or any(type(item) is not bytes for item in preimages_raw):
        _fail(FAIL_NODE_PREIMAGE, f"role {role.role_id} preimages are not byte strings")
    preimages = tuple(preimages_raw)
    if role.operation_id == OP_SEALED_SPLIT:
        if preimages or type(sealed_root) is not bytes or len(sealed_root) != 32:
            _fail(FAIL_NODE_PREIMAGE, f"sealed role {role.role_id} disclosure differs")
        return preimages, sealed_root
    if sealed_root is not None or len(preimages) != role.exact_count:
        _fail(FAIL_NODE_COUNT, f"role {role.role_id} preimage cardinality differs")
    return preimages, None


def _strict_prefixed_value(role: ReplayRoleSpecV1, payload: bytes) -> tuple[object, ...]:
    try:
        value = canonical_cbor_decode(payload)
    except Exception as exc:  # strict decoder has stable lower-level codes
        _fail(FAIL_NODE_PREIMAGE, f"role {role.role_id} has invalid canonical CBOR: {exc}")
    if (
        not isinstance(value, tuple)
        or len(value) != 3 + role.field_count
        or value[:3] != (1, role.tag, role.schema_id)
    ):
        _fail(FAIL_NODE_SCHEMA, f"role {role.role_id} preimage has a wrong formal prefix/length")
    return value


def _recompute_node_root(role: ReplayRoleSpecV1, preimages: tuple[bytes, ...], sealed: bytes | None) -> bytes:
    if role.operation_id == OP_SEALED_SPLIT:
        assert sealed is not None
        return sealed
    values = tuple(_strict_prefixed_value(role, item) for item in preimages)
    if role.operation_id == OP_CONTENT:
        assert role.domain is not None and len(values) == 1
        return content_hash(role.domain.decode("ascii"), values[0])
    if role.operation_id == OP_RFC6962:
        return rfc6962_root(values)
    _fail(FAIL_NODE_SCHEMA, f"unknown operation for role {role.role_id}")


def _typed_pair(
    universe_preimages: tuple[bytes, ...],
    truth_preimages: tuple[bytes, ...],
    *,
    signature_id: int,
    input_tag: int,
    input_schema: bytes,
) -> None:
    if len(universe_preimages) != len(truth_preimages):
        _fail(FAIL_TYPED_BINDING, "universe/truth row cardinality differs")
    for expected_index, (universe_raw, truth_raw) in enumerate(zip(universe_preimages, truth_preimages, strict=True)):
        universe = canonical_cbor_decode(universe_raw)
        truth = canonical_cbor_decode(truth_raw)
        if not isinstance(universe, tuple) or not isinstance(truth, tuple):
            _fail(FAIL_TYPED_BINDING, "typed row is not an array")
        if universe[3] != expected_index or truth[3] != expected_index or universe[4] != signature_id:
            _fail(FAIL_TYPED_BINDING, f"typed row index/signature differs at {expected_index}")
        nested = universe[5]
        if not isinstance(nested, tuple) or nested[:3] != (1, input_tag, input_schema):
            _fail(FAIL_ROLE_BINDING, f"typed input belongs to the wrong role at {expected_index}")
        if truth[4] != content_hash(CANONICAL_INPUT_DOMAIN, nested):
            _fail(FAIL_TYPED_BINDING, f"truth input hash differs at {expected_index}")


def _cross_bind_candidate(
    candidate: object,
    roots: Mapping[int, bytes],
    preimages: Mapping[int, tuple[bytes, ...]],
) -> None:
    assert hasattr(candidate, "fields")
    fields = candidate.fields
    for role in ROLE_SPECS:
        if fields[role.field_name] != roots[role.role_id]:
            _fail(FAIL_ROOT_BINDING, f"candidate field {role.field_name} differs from replay")

    for role_id, expected_role in ((10, 1), (11, 2)):
        decoded = decode_formal_object(preimages[role_id][0], expected_name="DslRoleBindingManifestV1")
        if decoded.fields["role_id"] != expected_role:
            _fail(FAIL_ROLE_BINDING, f"role-binding manifest {role_id} has the wrong role")
        for field_name, candidate_field in (
            ("child_dsl_spec_root", "child_dsl_spec_root"),
            ("child_freeze_root", "child_freeze_root"),
            ("operator_semantics_root", "operator_semantics_root"),
            ("identifier_registry_root", "identifier_registry_root"),
            ("canonical_ast_schema_root", "canonical_ast_schema_root"),
            ("canonical_cbor_profile_root", "canonical_cbor_profile_root"),
            ("split_binding_manifest_root", "split_binding_manifest_root"),
            ("custodian_binding_manifest_root", "custodian_binding_manifest_root"),
            ("seed_continuity_manifest_root", "seed_continuity_manifest_root"),
        ):
            if decoded.fields[field_name] != fields[candidate_field]:
                _fail(FAIL_ROLE_BINDING, f"{field_name} is spliced in role manifest {role_id}")
        universe_field = "outside_target_universe_root" if expected_role == 1 else "null_control_universe_root"
        truth_field = "outside_target_truth_root" if expected_role == 1 else "null_control_truth_root"
        if decoded.fields["formal_universe_root"] != fields[universe_field] or decoded.fields["formal_truth_root"] != fields[truth_field]:
            _fail(FAIL_ROLE_BINDING, f"typed roots are cross-role in role manifest {role_id}")

    split = decode_formal_object(preimages[12][0], expected_name="SplitBindingManifestV1").fields
    split_links = (
        ("outside_target_discovery_root", "outside_discovery_split_root"),
        ("outside_target_validation_root", "outside_validation_split_root"),
        ("outside_target_sealed_root", "outside_sealed_split_root"),
        ("null_control_discovery_root", "null_discovery_split_root"),
        ("null_control_validation_root", "null_validation_split_root"),
        ("null_control_sealed_root", "null_sealed_split_root"),
        ("hidden_access_ledger_genesis_root", "hidden_access_ledger_genesis_root"),
        ("hidden_access_ledger_head_root", "hidden_access_ledger_head_root"),
    )
    for source, target in split_links:
        if split[source] != fields[target]:
            _fail(FAIL_SPLIT_BINDING, f"split binding field {source} is spliced")

    _typed_pair(preimages[21], preimages[22], signature_id=1, input_tag=0x3401, input_schema=b"hegel-odd-input/1")
    _typed_pair(preimages[23], preimages[24], signature_id=2, input_tag=0x3402, input_schema=b"hegel-sink-input/1")


def _verify_purpose1_trust_and_signature(
    *,
    purpose_id: int,
    signature: object,
    key_manifest_raw: bytes,
    actor_trust_raw: bytes,
    bridge_root: bytes,
    created_at: int,
    repository_commit_id: object,
    signature_verifier: Ed25519VerifierV1 | None,
) -> bool:
    key = decode_formal_object(key_manifest_raw, expected_name="ActorKeyManifestV1")
    if key.fields["purpose_id"] != 1 or key.fields["repository_commit_id"] != repository_commit_id:
        _fail(FAIL_TRUST_BINDING, "purpose-1 key manifest purpose/commit differs")
    key_root = candidate_content_root("ActorKeyManifestV1", key.fields)
    trust = decode_formal_object(actor_trust_raw, expected_name="ActorTrustGenesisV1")
    entries = trust.fields["purpose_key_entries"]
    if not isinstance(entries, tuple) or tuple(item[0] for item in entries) != (1, 2, 3, 4):
        _fail(FAIL_TRUST_BINDING, "actor-trust purpose set differs")
    if entries[0][1] != key_root:
        _fail(FAIL_TRUST_BINDING, "purpose-1 key manifest is not bound by actor trust")
    public = _require_bytes(key.fields["public_key_32_bytes"], 32, FAIL_TRUST_BINDING, "purpose-1 public key")
    key_id = _require_bytes(key.fields["key_id"], 16, FAIL_TRUST_BINDING, "purpose-1 key ID")
    if key_id != hashlib.sha256(public).digest()[:16]:
        _fail(FAIL_TRUST_BINDING, "purpose-1 key ID does not derive from its public key")
    valid_until = key.fields["valid_until_unix_seconds_or_null"]
    if key.fields["valid_from_unix_seconds"] > created_at or (valid_until is not None and created_at > valid_until):
        _fail(FAIL_TRUST_BINDING, "purpose-1 key is outside its validity interval")

    if purpose_id == 1:
        if signature is not None:
            _fail(FAIL_SIGNATURE_PHASE, "purpose 1 accepts only an unsigned pre-sign package")
        return False
    signature_bytes = _require_bytes(signature, 64, FAIL_SIGNATURE, "purpose-1 bridge signature")
    if signature_verifier is None:
        _fail(FAIL_SIGNATURE, "no explicit dependency-free signature verifier was supplied")
    try:
        signature_verifier(
            public,
            signature_bytes,
            bridge_attestation_signature_preimage_v1(bridge_root, 1, key.fields["key_epoch"]),
        )
    except Exception as exc:
        _fail(FAIL_SIGNATURE, f"Ed25519 verification failed: {exc}")
    return True


def _write_private_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _require_exact_openssl_executable_v1(openssl_path: Path) -> Path:
    """Pin the sole qualified OpenSSL verifier executable by inode and bytes."""

    if (
        not isinstance(openssl_path, Path)
        or not openssl_path.is_absolute()
        or openssl_path != OPENSSL_EXECUTABLE
    ):
        _fail(
            FAIL_SIGNATURE,
            "OpenSSL verifier executable must be exactly /usr/bin/openssl",
        )
    descriptor: int | None = None
    try:
        lexical_before = openssl_path.lstat()
        if openssl_path.resolve(strict=True) != openssl_path:
            _fail(FAIL_SIGNATURE, "OpenSSL verifier executable path is not real")
        descriptor = os.open(
            openssl_path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        before = os.fstat(descriptor)
        if (
            stat.S_ISLNK(lexical_before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (lexical_before.st_dev, lexical_before.st_ino)
            != (before.st_dev, before.st_ino)
            or before.st_uid != 0
            or stat.S_IMODE(before.st_mode) != 0o755
            or before.st_size < 1
            or before.st_size > _MAX_OPENSSL_EXECUTABLE_BYTES
        ):
            _fail(FAIL_SIGNATURE, "OpenSSL verifier executable metadata differs")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                _fail(FAIL_SIGNATURE, "OpenSSL verifier executable read was short")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(FAIL_SIGNATURE, "OpenSSL verifier executable grew while read")
        after = os.fstat(descriptor)
        lexical_after = openssl_path.lstat()
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field)
            or getattr(after, field) != getattr(lexical_after, field)
            for field in stable_fields
        ):
            _fail(FAIL_SIGNATURE, "OpenSSL verifier executable changed while read")
        if digest.hexdigest() != OPENSSL_EXECUTABLE_SHA256:
            _fail(FAIL_SIGNATURE, "OpenSSL verifier executable SHA-256 differs")
        return openssl_path
    except BridgeDagReplayError:
        raise
    except OSError as exc:
        _fail(
            FAIL_SIGNATURE,
            f"OpenSSL verifier executable identity failed: {type(exc).__name__}",
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)


def make_openssl_ed25519_verifier_v1(
    private_temp_directory: Path,
    *,
    openssl_path: Path = OPENSSL_EXECUTABLE,
) -> Ed25519VerifierV1:
    """Return a no-network verifier using an explicit private directory.

    The directory must already exist as a non-symlink directory with mode
    ``0700``.  Files use O_EXCL/O_NOFOLLOW, mode ``0600``, fsync, an absolute
    OpenSSL executable, an empty stdin, a fixed environment, and finally-only
    cleanup.  Neither message nor signature bytes occur in argv.
    """

    directory = private_temp_directory.resolve(strict=True)
    metadata = directory.lstat()
    if (
        not private_temp_directory.is_absolute()
        or private_temp_directory.is_symlink()
        or directory != private_temp_directory
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        _fail(
            FAIL_SIGNATURE,
            "OpenSSL verifier directory must be an owned real mode-0700 directory",
        )
    executable = _require_exact_openssl_executable_v1(openssl_path)

    counter = 0

    def verify(public_key: bytes, signature: bytes, message: bytes) -> None:
        nonlocal counter
        _require_bytes(public_key, 32, FAIL_SIGNATURE, "Ed25519 public key")
        _require_bytes(signature, 64, FAIL_SIGNATURE, "Ed25519 signature")
        if type(message) is not bytes:
            _fail(FAIL_SIGNATURE, "Ed25519 message must be bytes")
        counter += 1
        stem = f"verify-{os.getpid()}-{counter}"
        public_path = directory / f"{stem}.public.der"
        message_path = directory / f"{stem}.message.bin"
        signature_path = directory / f"{stem}.signature.bin"
        paths = (public_path, message_path, signature_path)
        der = bytes.fromhex("302a300506032b6570032100") + public_key
        try:
            _write_private_exclusive(public_path, der)
            _write_private_exclusive(message_path, message)
            _write_private_exclusive(signature_path, signature)
            directory_descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            completed = subprocess.run(
                [
                    str(executable),
                    "pkeyutl",
                    "-verify",
                    "-pubin",
                    "-inkey",
                    str(public_path),
                    "-rawin",
                    "-in",
                    str(message_path),
                    "-sigfile",
                    str(signature_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    "LANG": "C",
                    "LC_ALL": "C",
                    "OPENSSL_CONF": "/dev/null",
                    "PATH": "/usr/bin:/bin",
                },
                timeout=30,
                check=False,
            )
            if completed.returncode != 0:
                _fail(FAIL_SIGNATURE, "OpenSSL rejected the Ed25519 signature")
        finally:
            for path in paths:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            directory_descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)

    return verify


def replay_bridge_dag_package_v1(
    package: bytes,
    *,
    allow_authoritative: bool = False,
    signature_verifier: Ed25519VerifierV1 | None = None,
) -> BridgeDagReplayResultV1:
    """Recompute the exact package and bridge DAG without signing anything."""

    try:
        value = canonical_cbor_decode(package)
    except Exception as exc:
        _fail(FAIL_PACKAGE_SCHEMA, f"package is not strict canonical CBOR: {exc}")
    if not isinstance(value, tuple) or len(value) != 12 or value[:3] != (1, PACKAGE_TAG, PACKAGE_SCHEMA_ID):
        _fail(FAIL_PACKAGE_SCHEMA, "package prefix/field count differs")
    _, _, _, authority, purpose_id, candidate_raw, bridge_raw, nodes_raw, key_raw, signature, created_at, commit = value
    if type(authority) is not bool:
        _fail(FAIL_PACKAGE_SCHEMA, "authority must be a CBOR boolean")
    if authority and not allow_authoritative:
        _fail(FAIL_PACKAGE_AUTHORITY, "authoritative replay requires the runtime-only opt in")
    if purpose_id not in (1, 2, 3):
        _fail(FAIL_PURPOSE, "purpose must be 1, 2, or 3")
    candidate_bytes = _require_bytes(candidate_raw, len(candidate_raw) if type(candidate_raw) is bytes else 0, FAIL_CANDIDATE, "candidate CBOR")
    bridge_bytes = _require_bytes(bridge_raw, len(bridge_raw) if type(bridge_raw) is bytes else 0, FAIL_BRIDGE, "bridge CBOR")
    key_bytes = _require_bytes(key_raw, len(key_raw) if type(key_raw) is bytes else 0, FAIL_TRUST_BINDING, "key manifest CBOR")
    if not isinstance(nodes_raw, tuple) or len(nodes_raw) != len(ROLE_SPECS):
        _fail(FAIL_NODE_SET, "replay node count differs")

    roots: dict[int, bytes] = {}
    preimages: dict[int, tuple[bytes, ...]] = {}
    for spec, raw in zip(ROLE_SPECS, nodes_raw, strict=True):
        decoded_preimages, sealed = _decode_node(spec, raw)
        roots[spec.role_id] = _recompute_node_root(spec, decoded_preimages, sealed)
        preimages[spec.role_id] = decoded_preimages

    try:
        candidate = decode_formal_object(candidate_bytes, expected_name="M3ExecutionCandidateV1")
    except Exception as exc:
        _fail(FAIL_CANDIDATE, str(exc))
    if candidate.fields["created_at_unix_seconds"] != created_at or candidate.fields["repository_commit_id"] != commit:
        _fail(FAIL_CANDIDATE, "package time/commit differs from candidate")
    _cross_bind_candidate(candidate, MappingProxyType(roots), MappingProxyType(preimages))
    candidate_root = candidate_content_root("M3ExecutionCandidateV1", candidate.fields)

    try:
        bridge = decode_formal_object(bridge_bytes, expected_name="BridgeReplayStatementV1")
    except Exception as exc:
        _fail(FAIL_BRIDGE, str(exc))
    bridge_expected = {
        "run_id": candidate.fields["run_id"],
        "diagnostic_formal_bridge_root": candidate.fields["diagnostic_formal_bridge_root"],
        "m3_execution_candidate_root": candidate_root,
        "child_dsl_spec_root": candidate.fields["child_dsl_spec_root"],
        "child_freeze_root": candidate.fields["child_freeze_root"],
        "actor_trust_genesis_root": candidate.fields["actor_trust_genesis_root"],
        "opaque_id_registry_snapshot_root": candidate.fields["opaque_id_registry_snapshot_root"],
    }
    if dict(bridge.fields) != bridge_expected:
        _fail(FAIL_BRIDGE, "bridge statement does not exactly project the candidate")
    bridge_root = candidate_content_root("BridgeReplayStatementV1", bridge.fields)
    signature_verified = _verify_purpose1_trust_and_signature(
        purpose_id=purpose_id,
        signature=signature,
        key_manifest_raw=key_bytes,
        actor_trust_raw=preimages[20][0],
        bridge_root=bridge_root,
        created_at=created_at,
        repository_commit_id=commit,
        signature_verifier=signature_verifier,
    )
    return BridgeDagReplayResultV1(
        package_digest=content_hash(PACKAGE_HASH_DOMAIN, value),
        candidate_root=candidate_root,
        bridge_statement_root=bridge_root,
        purpose_id=purpose_id,
        purpose1_signature_verified=signature_verified,
        eligible_to_sign_bridge_statement=purpose_id in (1, 2, 3),
        authoritative=authority,
    )


__all__ = [
    "ACTOR_REPLAY_IMPLEMENTATIONS",
    "ACTOR_REPLAY_RECEIPT_SCHEMA",
    "BridgeDagReplayError",
    "BridgeDagReplayResultV1",
    "FAIL_BRIDGE",
    "FAIL_ACTOR_RECEIPT",
    "FAIL_CANDIDATE",
    "FAIL_NODE_COUNT",
    "FAIL_NODE_PREIMAGE",
    "FAIL_NODE_SCHEMA",
    "FAIL_NODE_SET",
    "FAIL_PACKAGE_AUTHORITY",
    "FAIL_PACKAGE_SCHEMA",
    "FAIL_PURPOSE",
    "FAIL_ROLE_BINDING",
    "FAIL_ROOT_BINDING",
    "FAIL_SIGNATURE",
    "FAIL_SIGNATURE_PHASE",
    "FAIL_SPLIT_BINDING",
    "FAIL_TRUST_BINDING",
    "FAIL_TYPED_BINDING",
    "OP_CONTENT",
    "OP_RFC6962",
    "OP_SEALED_SPLIT",
    "PACKAGE_HASH_DOMAIN",
    "PACKAGE_SCHEMA_ID",
    "PACKAGE_TAG",
    "ReplayNodeV1",
    "ReplayRoleSpecV1",
    "ROLE_SPECS",
    "build_bridge_dag_replay_package_v1",
    "build_bridge_actor_replay_receipt_v1",
    "make_openssl_ed25519_verifier_v1",
    "replay_bridge_dag_package_v1",
    "validate_bridge_actor_replay_receipt_v1",
]
