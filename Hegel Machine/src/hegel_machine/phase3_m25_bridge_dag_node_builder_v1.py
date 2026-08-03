"""Deterministic construction of the complete M2.5 bridge replay DAG.

This module is deliberately a public-data assembler.  It turns one qualified
``FormalStaticBasisV1`` plus the ceremony's already-created public formal
fields into the exact 37 ``ReplayNodeV1`` values consumed by
``phase3_m25_bridge_full_dag_replay_v1``.  It does not create entropy, keys,
signatures, authoritative formal roots, or an M3 state transition.

The builder is intentionally stricter than a plain serializer:

* every input map has an exact, frozen role-name set;
* every public preimage is encoded through the formal schema registry;
* every record-tree role has its frozen cardinality and ordering;
* static, typed, and M3 implementation/contract roots must agree with the
  qualified basis as well as the execution candidate;
* dynamic cross-links are checked before any package is returned; and
* package construction is preflighted as an unsigned purpose-1 replay.  This
  validates the whole public DAG and purpose-1 trust binding without treating
  an unverified signature as valid.

Purpose-2/3 package construction accepts an already-produced purpose-1
signature.  Cryptographic verification remains the responsibility of the
isolated purpose-2/3 replayer and is not simulated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping, NoReturn, Sequence

from .phase3_m25_bridge_full_dag_replay_v1 import (
    BridgeDagReplayError,
    OP_CONTENT,
    OP_RFC6962,
    OP_SEALED_SPLIT,
    ROLE_SPECS,
    ReplayNodeV1,
    build_bridge_dag_replay_package_v1,
    replay_bridge_dag_package_v1,
)
from .phase3_m25_formal_static_basis_v1 import FormalStaticBasisV1
from .phase3_m25_rows_v1 import TypedRoleRows
from .phase3_m25_wire_v1 import (
    candidate_content_root,
    candidate_record_tree_root,
    decode_formal_object,
    encode_formal_object,
)
from .strict_cbor_v1 import canonical_cbor_encode, rfc6962_root


FAIL_INPUT_TYPE: Final = "FAIL_M25_BRIDGE_NODE_BUILD_INPUT_TYPE"
FAIL_ROLE_LAYOUT: Final = "FAIL_M25_BRIDGE_NODE_BUILD_ROLE_LAYOUT"
FAIL_FIELD_SET: Final = "FAIL_M25_BRIDGE_NODE_BUILD_FIELD_SET"
FAIL_PREIMAGE: Final = "FAIL_M25_BRIDGE_NODE_BUILD_PREIMAGE"
FAIL_COUNT: Final = "FAIL_M25_BRIDGE_NODE_BUILD_COUNT"
FAIL_BASIS_BINDING: Final = "FAIL_M25_BRIDGE_NODE_BUILD_BASIS_BINDING"
FAIL_CANDIDATE_BINDING: Final = "FAIL_M25_BRIDGE_NODE_BUILD_CANDIDATE_BINDING"
FAIL_CROSS_ROLE: Final = "FAIL_M25_BRIDGE_NODE_BUILD_CROSS_ROLE"
FAIL_SIGNATURE_PHASE: Final = "FAIL_M25_BRIDGE_NODE_BUILD_SIGNATURE_PHASE"
FAIL_PACKAGE_PREFLIGHT: Final = "FAIL_M25_BRIDGE_NODE_BUILD_PACKAGE_PREFLIGHT"


class BridgeDagNodeBuildError(RuntimeError):
    """Stable, fail-closed error raised before actor package replay."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise BridgeDagNodeBuildError(code, detail)


# This duplicated layout is an intentional drift tripwire.  A change to the
# underlying replay protocol must update the builder explicitly; silently
# inheriting a 38th role or a changed count would weaken the bridge boundary.
_EXPECTED_ROLE_LAYOUT: Final = (
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


DYNAMIC_OBJECT_SCHEMAS: Final = MappingProxyType(
    {
        "shrink_transition_root": "DslShrinkTransitionFormalV1",
        "outside_target_binding_manifest_root": "DslRoleBindingManifestV1",
        "null_control_binding_manifest_root": "DslRoleBindingManifestV1",
        "split_binding_manifest_root": "SplitBindingManifestV1",
        "custodian_binding_manifest_root": "CustodianBindingManifestV1",
        "seed_continuity_manifest_root": "SeedContinuityManifestV1",
        "hidden_access_ledger_genesis_root": "HiddenAccessLedgerRecordV1",
        "hidden_access_ledger_head_root": "HiddenAccessLedgerRecordV1",
    }
)

SEALED_SPLIT_FIELD_NAMES: Final = (
    "outside_discovery_split_root",
    "outside_validation_split_root",
    "outside_sealed_split_root",
    "null_discovery_split_root",
    "null_validation_split_root",
    "null_sealed_split_root",
)


@dataclass(frozen=True, slots=True)
class M3ExecutionBindingContractFieldsV1:
    """Exact field maps for candidate roles 31--37."""

    python_implementation_binding_fields: Mapping[str, object]
    rust_implementation_binding_fields: Mapping[str, object]
    traversal_contract_fields: Mapping[str, object]
    bucket_accounting_contract_fields: Mapping[str, object]
    program_archive_contract_fields: Mapping[str, object]
    output_archive_contract_fields: Mapping[str, object]
    state_machine_contract_fields: Mapping[str, object]

    def by_candidate_field(self) -> Mapping[str, tuple[str, Mapping[str, object]]]:
        return MappingProxyType(
            {
                "python_implementation_binding_root": (
                    "ImplementationBindingV1",
                    self.python_implementation_binding_fields,
                ),
                "rust_implementation_binding_root": (
                    "ImplementationBindingV1",
                    self.rust_implementation_binding_fields,
                ),
                "traversal_contract_root": (
                    "TraversalContractV1",
                    self.traversal_contract_fields,
                ),
                "bucket_accounting_contract_root": (
                    "BucketAccountingContractV1",
                    self.bucket_accounting_contract_fields,
                ),
                "program_archive_contract_root": (
                    "ProgramArchiveContractV1",
                    self.program_archive_contract_fields,
                ),
                "output_archive_contract_root": (
                    "OutputArchiveContractV1",
                    self.output_archive_contract_fields,
                ),
                "state_machine_contract_root": (
                    "StateMachineContractV1",
                    self.state_machine_contract_fields,
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class BridgeDagNodeBuildInputsV1:
    """Complete public inputs needed to materialize replay roles 1--37."""

    basis: FormalStaticBasisV1
    candidate_fields: Mapping[str, object]
    dynamic_object_fields: Mapping[str, Mapping[str, object]]
    external_attestation_bundle_fields: Mapping[str, object]
    parent_attestation_fields: Mapping[str, object]
    final_opaque_snapshot_fields: Mapping[str, object]
    actor_trust_fields: Mapping[str, object]
    outside_typed_rows: TypedRoleRows
    null_typed_rows: TypedRoleRows
    sealed_split_roots: Mapping[str, bytes]
    m3_execution_fields: M3ExecutionBindingContractFieldsV1


@dataclass(frozen=True, slots=True)
class BridgeDagPackageBuildInputsV1:
    """Public package inputs; an existing signature is opaque input only."""

    node_inputs: BridgeDagNodeBuildInputsV1
    purpose_id: int
    bridge_statement_fields: Mapping[str, object]
    purpose1_actor_key_manifest_fields: Mapping[str, object]
    purpose1_bridge_signature: bytes | None
    authority: bool = False


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(FAIL_INPUT_TYPE, f"{label} must be a mapping")
    if any(type(key) is not str for key in value):
        _fail(FAIL_FIELD_SET, f"{label} contains a non-text key")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: Sequence[str], label: str) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        _fail(FAIL_FIELD_SET, f"{label} role names differ; missing={missing}, extra={extra}")


def _require_bytes32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        _fail(FAIL_PREIMAGE, f"{label} must be exactly 32 bytes")
    return value


def _require_role_layout_v1() -> None:
    actual = tuple(
        (
            spec.role_id,
            spec.candidate_index,
            spec.field_name,
            spec.operation_id,
            spec.schema_name,
            spec.exact_count,
        )
        for spec in ROLE_SPECS
    )
    if actual != _EXPECTED_ROLE_LAYOUT:
        _fail(FAIL_ROLE_LAYOUT, "bridge replay ROLE_SPECS differs from the frozen 37-role layout")


def _formal_preimage(schema_name: str, fields: Mapping[str, object], label: str) -> bytes:
    _require_mapping(fields, label)
    try:
        payload = encode_formal_object(schema_name, fields)
        decoded = decode_formal_object(payload, expected_name=schema_name)
        if encode_formal_object(schema_name, decoded.fields) != payload:
            _fail(FAIL_PREIMAGE, f"{label} did not round-trip byte exactly")
        return payload
    except BridgeDagNodeBuildError:
        raise
    except Exception as exc:
        _fail(FAIL_PREIMAGE, f"{label} is not an exact {schema_name} preimage: {exc}")


def _content_node(
    role_id: int,
    schema_name: str,
    fields: Mapping[str, object],
    label: str,
) -> tuple[ReplayNodeV1, bytes]:
    payload = _formal_preimage(schema_name, fields, label)
    try:
        root = candidate_content_root(schema_name, fields)
    except Exception as exc:
        _fail(FAIL_PREIMAGE, f"{label} content root failed: {exc}")
    return ReplayNodeV1(role_id, (payload,)), root


def _record_node(
    role_id: int,
    schema_name: str,
    rows: Sequence[Mapping[str, object]],
    expected_count: int,
    label: str,
) -> tuple[ReplayNodeV1, bytes]:
    if not isinstance(rows, (tuple, list)) or len(rows) != expected_count:
        _fail(FAIL_COUNT, f"{label} must contain exactly {expected_count} rows")
    preimages = tuple(
        _formal_preimage(schema_name, _require_mapping(row, f"{label}[{index}]"), f"{label}[{index}]")
        for index, row in enumerate(rows)
    )
    try:
        root = candidate_record_tree_root(schema_name, rows)
    except Exception as exc:
        _fail(FAIL_PREIMAGE, f"{label} record root failed: {exc}")
    return ReplayNodeV1(role_id, preimages), root


def _typed_nodes(
    universe_role_id: int,
    truth_role_id: int,
    rows: TypedRoleRows,
    *,
    expected_role_name: str,
    expected_signature_id: int,
    expected_count: int,
) -> tuple[tuple[ReplayNodeV1, bytes], tuple[ReplayNodeV1, bytes]]:
    if not isinstance(rows, TypedRoleRows):
        _fail(FAIL_INPUT_TYPE, f"{expected_role_name} typed rows must be TypedRoleRows")
    if rows.role_name != expected_role_name or rows.input_signature_id != expected_signature_id:
        _fail(FAIL_CROSS_ROLE, f"{expected_role_name} typed-row role/signature identity differs")
    if len(rows.universe_rows) != expected_count or len(rows.truth_rows) != expected_count:
        _fail(FAIL_COUNT, f"{expected_role_name} typed rows must contain exactly {expected_count} pairs")
    try:
        rows.validate()
    except Exception as exc:
        _fail(FAIL_CROSS_ROLE, f"{expected_role_name} typed rows failed role validation: {exc}")

    def encode_rows(schema_name: str, values: Sequence[tuple[object, ...]]) -> tuple[bytes, ...]:
        encoded: list[bytes] = []
        for index, value in enumerate(values):
            try:
                payload = canonical_cbor_encode(value)
                decoded = decode_formal_object(payload, expected_name=schema_name)
                if encode_formal_object(schema_name, decoded.fields) != payload:
                    _fail(FAIL_PREIMAGE, f"{expected_role_name} {schema_name}[{index}] round-trip differs")
            except BridgeDagNodeBuildError:
                raise
            except Exception as exc:
                _fail(FAIL_PREIMAGE, f"{expected_role_name} {schema_name}[{index}] is invalid: {exc}")
            encoded.append(payload)
        return tuple(encoded)

    universe_preimages = encode_rows("BoundedUniverseRowV1", rows.universe_rows)
    truth_preimages = encode_rows("TargetTruthRowV1", rows.truth_rows)
    universe_root = rfc6962_root(rows.universe_rows)
    truth_root = rfc6962_root(rows.truth_rows)
    return (
        (ReplayNodeV1(universe_role_id, universe_preimages), universe_root),
        (ReplayNodeV1(truth_role_id, truth_preimages), truth_root),
    )


def _require_candidate_crosslinks_v1(inputs: BridgeDagNodeBuildInputsV1) -> None:
    candidate = inputs.candidate_fields
    dynamic = inputs.dynamic_object_fields
    commit = candidate["repository_commit_id"]
    timestamp = candidate["created_at_unix_seconds"]

    # All ceremony-created child objects share the candidate identity.  The
    # parent attestation intentionally binds the frozen parent commit instead.
    for field_name in (
        "shrink_transition_root",
        "outside_target_binding_manifest_root",
        "null_control_binding_manifest_root",
        "split_binding_manifest_root",
        "custodian_binding_manifest_root",
        "seed_continuity_manifest_root",
        "hidden_access_ledger_genesis_root",
        "hidden_access_ledger_head_root",
    ):
        fields = dynamic[field_name]
        if fields.get("repository_commit_id") != commit:
            _fail(FAIL_CANDIDATE_BINDING, f"{field_name} repository commit differs from candidate")
    for field_name in (
        "shrink_transition_root",
        "outside_target_binding_manifest_root",
        "null_control_binding_manifest_root",
        "split_binding_manifest_root",
        "hidden_access_ledger_genesis_root",
        "hidden_access_ledger_head_root",
    ):
        if dynamic[field_name].get("created_at_unix_seconds") != timestamp:
            _fail(FAIL_CANDIDATE_BINDING, f"{field_name} timestamp differs from candidate")
    if inputs.actor_trust_fields.get("repository_commit_id") != commit or inputs.actor_trust_fields.get(
        "created_at_unix_seconds"
    ) != timestamp:
        _fail(FAIL_CANDIDATE_BINDING, "actor trust identity differs from candidate")
    if inputs.final_opaque_snapshot_fields.get("repository_commit_id") != commit:
        _fail(FAIL_CANDIDATE_BINDING, "opaque snapshot commit differs from candidate")

    genesis = dynamic["hidden_access_ledger_genesis_root"]
    head = dynamic["hidden_access_ledger_head_root"]
    if dict(genesis) != dict(head):
        _fail(FAIL_CROSS_ROLE, "pre-M3 ledger head must be the exact genesis record preimage")

    continuity = dynamic["seed_continuity_manifest_root"]
    if (
        continuity.get("parent_manifest_absence_attestation_root")
        != candidate["parent_absence_attestation_root"]
        or continuity.get("hidden_access_ledger_genesis_root")
        != candidate["hidden_access_ledger_genesis_root"]
    ):
        _fail(FAIL_CROSS_ROLE, "seed continuity parent/ledger linkage differs")

    custodian = dynamic["custodian_binding_manifest_root"]
    if (
        custodian.get("hidden_access_ledger_genesis_root")
        != candidate["hidden_access_ledger_genesis_root"]
        or custodian.get("seed_continuity_manifest_root")
        != candidate["seed_continuity_manifest_root"]
    ):
        _fail(FAIL_CROSS_ROLE, "custodian continuity/ledger linkage differs")

    split = dynamic["split_binding_manifest_root"]
    split_links = {
        "seed_continuity_manifest_root": "seed_continuity_manifest_root",
        "outside_target_discovery_root": "outside_discovery_split_root",
        "outside_target_validation_root": "outside_validation_split_root",
        "outside_target_sealed_root": "outside_sealed_split_root",
        "null_control_discovery_root": "null_discovery_split_root",
        "null_control_validation_root": "null_validation_split_root",
        "null_control_sealed_root": "null_sealed_split_root",
        "hidden_access_ledger_genesis_root": "hidden_access_ledger_genesis_root",
        "hidden_access_ledger_head_root": "hidden_access_ledger_head_root",
    }
    for source, target in split_links.items():
        if split.get(source) != candidate[target]:
            _fail(FAIL_CROSS_ROLE, f"split field {source} is spliced")

    for candidate_field, expected_role, universe_field, truth_field in (
        (
            "outside_target_binding_manifest_root",
            1,
            "outside_target_universe_root",
            "outside_target_truth_root",
        ),
        (
            "null_control_binding_manifest_root",
            2,
            "null_control_universe_root",
            "null_control_truth_root",
        ),
    ):
        role = dynamic[candidate_field]
        expected = {
            "role_id": expected_role,
            "child_dsl_spec_root": candidate["child_dsl_spec_root"],
            "child_freeze_root": candidate["child_freeze_root"],
            "operator_semantics_root": candidate["operator_semantics_root"],
            "identifier_registry_root": candidate["identifier_registry_root"],
            "canonical_ast_schema_root": candidate["canonical_ast_schema_root"],
            "canonical_cbor_profile_root": candidate["canonical_cbor_profile_root"],
            "formal_universe_root": candidate[universe_field],
            "formal_truth_root": candidate[truth_field],
            "split_binding_manifest_root": candidate["split_binding_manifest_root"],
            "custodian_binding_manifest_root": candidate["custodian_binding_manifest_root"],
            "seed_continuity_manifest_root": candidate["seed_continuity_manifest_root"],
            "parent_manifest_absence_attestation_root_or_null": candidate[
                "parent_absence_attestation_root"
            ],
        }
        for field, value in expected.items():
            if role.get(field) != value:
                _fail(FAIL_CROSS_ROLE, f"{candidate_field}.{field} is cross-role or spliced")

    shrink = dynamic["shrink_transition_root"]
    shrink_links = {
        "child_dsl_spec_root": "child_dsl_spec_root",
        "child_freeze_root": "child_freeze_root",
        "approval_manifest_root": "approval_manifest_root",
        "outside_target_binding_manifest_root": "outside_target_binding_manifest_root",
        "null_control_binding_manifest_root": "null_control_binding_manifest_root",
        "split_binding_manifest_root": "split_binding_manifest_root",
        "custodian_binding_manifest_root": "custodian_binding_manifest_root",
        "seed_continuity_manifest_root": "seed_continuity_manifest_root",
    }
    for source, target in shrink_links.items():
        if shrink.get(source) != candidate[target]:
            _fail(FAIL_CROSS_ROLE, f"shrink transition field {source} is spliced")

    for candidate_field, (schema_name, fields) in inputs.m3_execution_fields.by_candidate_field().items():
        if schema_name == "ImplementationBindingV1":
            expected_id = 1 if candidate_field.startswith("python_") else 2
            if fields.get("implementation_id") != expected_id:
                _fail(FAIL_CROSS_ROLE, f"{candidate_field} implementation identity differs")
            if fields.get("repository_commit_id") != commit:
                _fail(FAIL_CANDIDATE_BINDING, f"{candidate_field} commit differs from candidate")


def build_bridge_dag_nodes_v1(inputs: BridgeDagNodeBuildInputsV1) -> tuple[ReplayNodeV1, ...]:
    """Build and fully root-check the frozen role-1..37 node sequence."""

    if not isinstance(inputs, BridgeDagNodeBuildInputsV1):
        _fail(FAIL_INPUT_TYPE, "inputs must be BridgeDagNodeBuildInputsV1")
    if not isinstance(inputs.basis, FormalStaticBasisV1):
        _fail(FAIL_INPUT_TYPE, "basis must be FormalStaticBasisV1")
    _require_role_layout_v1()
    candidate = _require_mapping(inputs.candidate_fields, "candidate_fields")
    dynamic_raw = _require_mapping(inputs.dynamic_object_fields, "dynamic_object_fields")
    _require_exact_keys(dynamic_raw, tuple(DYNAMIC_OBJECT_SCHEMAS), "dynamic_object_fields")
    dynamic: Mapping[str, Mapping[str, object]] = MappingProxyType(
        {
            name: _require_mapping(dynamic_raw[name], f"dynamic_object_fields[{name}]")
            for name in DYNAMIC_OBJECT_SCHEMAS
        }
    )
    sealed_raw = _require_mapping(inputs.sealed_split_roots, "sealed_split_roots")
    _require_exact_keys(sealed_raw, SEALED_SPLIT_FIELD_NAMES, "sealed_split_roots")
    sealed = MappingProxyType(
        {name: _require_bytes32(sealed_raw[name], f"sealed_split_roots[{name}]") for name in SEALED_SPLIT_FIELD_NAMES}
    )
    if not isinstance(inputs.m3_execution_fields, M3ExecutionBindingContractFieldsV1):
        _fail(FAIL_INPUT_TYPE, "m3_execution_fields must be M3ExecutionBindingContractFieldsV1")

    try:
        # Exact schema/field-set validation, including the frozen 44-field
        # candidate shape and the pre-M3 ledger-head guard.
        encode_formal_object("M3ExecutionCandidateV1", candidate)
    except Exception as exc:
        _fail(FAIL_PREIMAGE, f"candidate_fields is not an exact M3ExecutionCandidateV1: {exc}")

    for field, expected in inputs.basis.m3_candidate_static_fields.items():
        if candidate.get(field) != expected:
            _fail(FAIL_BASIS_BINDING, f"candidate static field {field} differs from qualified basis")

    nodes: dict[int, ReplayNodeV1] = {}
    roots: dict[int, bytes] = {}

    def add(role_id: int, built: tuple[ReplayNodeV1, bytes]) -> None:
        if role_id in nodes or built[0].role_id != role_id:
            _fail(FAIL_ROLE_LAYOUT, f"duplicate or mislabelled role {role_id}")
        nodes[role_id], roots[role_id] = built

    static_content = (
        (1, "DslSpecV1", "child_dsl_spec", "child_dsl_spec_root"),
        (2, "FreezeSpecV1", "child_freeze", "child_freeze_root"),
        (3, "NormativeApprovalManifestV1", "normative_approval_manifest", "approval_manifest_root"),
        (7, "CanonicalAstProfileSpecV1", "canonical_ast_profile", "canonical_ast_schema_root"),
        (8, "CanonicalCborProfileSpecV1", "canonical_cbor_profile", "canonical_cbor_profile_root"),
    )
    for role_id, schema_name, object_name, root_name in static_content:
        try:
            fields = inputs.basis.objects[object_name]
            expected_basis_root = inputs.basis.roots[root_name]
        except KeyError:
            _fail(FAIL_BASIS_BINDING, f"basis omits {object_name}/{root_name}")
        built = _content_node(role_id, schema_name, fields, f"basis.objects[{object_name}]")
        if built[1] != expected_basis_root:
            _fail(FAIL_BASIS_BINDING, f"basis root {root_name} differs from its exact preimage")
        add(role_id, built)

    static_records = (
        (5, "OperatorSemanticsEntryV1", "operator_semantics", "operator_semantics_root", 28),
        (6, "IdentifierRegistryEntryV1", "identifier_registry", "identifier_registry_root", 55),
        (9, "DiagnosticFormalBridgeRecordV1", "diagnostic_formal_bridge", "diagnostic_formal_bridge_root", 12),
    )
    for role_id, schema_name, record_name, root_name, count in static_records:
        try:
            rows = inputs.basis.record_sets[record_name]
            expected_basis_root = inputs.basis.roots[root_name]
        except KeyError:
            _fail(FAIL_BASIS_BINDING, f"basis omits {record_name}/{root_name}")
        built = _record_node(role_id, schema_name, rows, count, f"basis.record_sets[{record_name}]")
        if built[1] != expected_basis_root:
            _fail(FAIL_BASIS_BINDING, f"basis root {root_name} differs from its exact rows")
        add(role_id, built)

    dynamic_role_ids = {
        "shrink_transition_root": 4,
        "outside_target_binding_manifest_root": 10,
        "null_control_binding_manifest_root": 11,
        "split_binding_manifest_root": 12,
        "custodian_binding_manifest_root": 13,
        "seed_continuity_manifest_root": 14,
        "hidden_access_ledger_genesis_root": 17,
        "hidden_access_ledger_head_root": 18,
    }
    for field_name, schema_name in DYNAMIC_OBJECT_SCHEMAS.items():
        add(
            dynamic_role_ids[field_name],
            _content_node(
                dynamic_role_ids[field_name],
                schema_name,
                dynamic[field_name],
                f"dynamic_object_fields[{field_name}]",
            ),
        )

    add(
        15,
        _content_node(
            15,
            "AttestationBundleV1",
            inputs.external_attestation_bundle_fields,
            "external_attestation_bundle_fields",
        ),
    )
    add(
        16,
        _content_node(
            16,
            "ParentManifestAbsenceAttestationV2",
            inputs.parent_attestation_fields,
            "parent_attestation_fields",
        ),
    )
    add(
        19,
        _content_node(
            19,
            "OpaqueIdRegistrySnapshotV1",
            inputs.final_opaque_snapshot_fields,
            "final_opaque_snapshot_fields",
        ),
    )
    add(
        20,
        _content_node(20, "ActorTrustGenesisV1", inputs.actor_trust_fields, "actor_trust_fields"),
    )

    odd_nodes = _typed_nodes(
        21,
        22,
        inputs.outside_typed_rows,
        expected_role_name="odd",
        expected_signature_id=1,
        expected_count=480,
    )
    sink_nodes = _typed_nodes(
        23,
        24,
        inputs.null_typed_rows,
        expected_role_name="sink",
        expected_signature_id=2,
        expected_count=85,
    )
    add(21, odd_nodes[0])
    add(22, odd_nodes[1])
    add(23, sink_nodes[0])
    add(24, sink_nodes[1])

    sealed_role_ids = dict(zip(SEALED_SPLIT_FIELD_NAMES, range(25, 31), strict=True))
    for field_name in SEALED_SPLIT_FIELD_NAMES:
        role_id = sealed_role_ids[field_name]
        add(role_id, (ReplayNodeV1(role_id, sealed_root=sealed[field_name]), sealed[field_name]))

    m3_role_ids = {
        "python_implementation_binding_root": 31,
        "rust_implementation_binding_root": 32,
        "traversal_contract_root": 33,
        "bucket_accounting_contract_root": 34,
        "program_archive_contract_root": 35,
        "output_archive_contract_root": 36,
        "state_machine_contract_root": 37,
    }
    m3_basis_object_names = {
        "python_implementation_binding_root": "python_m3_implementation_binding",
        "rust_implementation_binding_root": "rust_m3_implementation_binding",
        "traversal_contract_root": "traversal_contract",
        "bucket_accounting_contract_root": "bucket_accounting_contract",
        "program_archive_contract_root": "program_archive_contract",
        "output_archive_contract_root": "output_archive_contract",
        "state_machine_contract_root": "state_machine_contract",
    }
    for field_name, (schema_name, fields) in inputs.m3_execution_fields.by_candidate_field().items():
        role_id = m3_role_ids[field_name]
        expected_basis_fields = inputs.basis.objects.get(m3_basis_object_names[field_name])
        if not isinstance(expected_basis_fields, Mapping) or dict(fields) != dict(expected_basis_fields):
            _fail(FAIL_BASIS_BINDING, f"qualified basis exact preimage differs for {field_name}")
        built = _content_node(role_id, schema_name, fields, f"m3_execution_fields[{field_name}]")
        expected_basis_root = inputs.basis.roots.get(field_name)
        if type(expected_basis_root) is not bytes or built[1] != expected_basis_root:
            _fail(FAIL_BASIS_BINDING, f"qualified basis does not bind exact {field_name} preimage")
        add(role_id, built)

    expected_ids = tuple(range(1, 38))
    if tuple(sorted(nodes)) != expected_ids:
        _fail(FAIL_ROLE_LAYOUT, "constructed replay role IDs are not exactly 1..37")

    for spec in ROLE_SPECS:
        root = roots[spec.role_id]
        candidate_root = candidate.get(spec.field_name)
        if type(candidate_root) is not bytes or len(candidate_root) != 32:
            _fail(FAIL_CANDIDATE_BINDING, f"candidate field {spec.field_name} is not a 32-byte root")
        if root != candidate_root:
            _fail(FAIL_CANDIDATE_BINDING, f"candidate field {spec.field_name} differs from role {spec.role_id}")
        node = nodes[spec.role_id]
        if spec.operation_id == OP_SEALED_SPLIT:
            if node.preimages or node.sealed_root != root:
                _fail(FAIL_ROLE_LAYOUT, f"sealed role {spec.role_id} disclosure differs")
        elif len(node.preimages) != spec.exact_count or node.sealed_root is not None:
            _fail(FAIL_COUNT, f"role {spec.role_id} preimage count differs from {spec.exact_count}")

    # The typed rows are part of the frozen static basis, not merely a set of
    # candidate-controlled roots.
    for role_id in (21, 22, 23, 24):
        root_name = ROLE_SPECS[role_id - 1].field_name
        if inputs.basis.roots.get(root_name) != roots[role_id]:
            _fail(FAIL_BASIS_BINDING, f"typed role {root_name} differs from the static basis")

    # Use the normalized dynamic map for the semantic cross-link checks.
    normalized_inputs = BridgeDagNodeBuildInputsV1(
        basis=inputs.basis,
        candidate_fields=candidate,
        dynamic_object_fields=dynamic,
        external_attestation_bundle_fields=inputs.external_attestation_bundle_fields,
        parent_attestation_fields=inputs.parent_attestation_fields,
        final_opaque_snapshot_fields=inputs.final_opaque_snapshot_fields,
        actor_trust_fields=inputs.actor_trust_fields,
        outside_typed_rows=inputs.outside_typed_rows,
        null_typed_rows=inputs.null_typed_rows,
        sealed_split_roots=sealed,
        m3_execution_fields=inputs.m3_execution_fields,
    )
    _require_candidate_crosslinks_v1(normalized_inputs)
    return tuple(nodes[role_id] for role_id in expected_ids)


def build_bridge_dag_replay_package_from_inputs_v1(
    inputs: BridgeDagPackageBuildInputsV1,
) -> bytes:
    """Build a role-complete package without creating or faking a signature.

    The DAG is first replayed as a non-authoritative, unsigned purpose-1
    package.  For purposes 2 and 3, the supplied 64-byte signature is then
    carried verbatim into the returned package; its Ed25519 verification must
    occur inside the receiving isolated actor.
    """

    if not isinstance(inputs, BridgeDagPackageBuildInputsV1):
        _fail(FAIL_INPUT_TYPE, "inputs must be BridgeDagPackageBuildInputsV1")
    if type(inputs.purpose_id) is not int or inputs.purpose_id not in (1, 2, 3):
        _fail(FAIL_SIGNATURE_PHASE, "purpose_id must be CBOR uint 1, 2, or 3")
    if type(inputs.authority) is not bool:
        _fail(FAIL_INPUT_TYPE, "authority must be a boolean")
    if inputs.purpose_id == 1:
        if inputs.purpose1_bridge_signature is not None:
            _fail(FAIL_SIGNATURE_PHASE, "purpose 1 package must be unsigned")
    elif type(inputs.purpose1_bridge_signature) is not bytes or len(
        inputs.purpose1_bridge_signature
    ) != 64:
        _fail(FAIL_SIGNATURE_PHASE, "purpose 2/3 package requires one 64-byte purpose-1 signature")

    nodes = build_bridge_dag_nodes_v1(inputs.node_inputs)
    key_fields = _require_mapping(
        inputs.purpose1_actor_key_manifest_fields,
        "purpose1_actor_key_manifest_fields",
    )
    bridge_fields = _require_mapping(inputs.bridge_statement_fields, "bridge_statement_fields")

    # This is a structural/trust preflight only.  It deliberately uses the
    # unsigned purpose-1 phase so no callback can accidentally bless an
    # unverified purpose-1 signature for a purpose-2/3 package.
    try:
        probe = build_bridge_dag_replay_package_v1(
            purpose_id=1,
            candidate_fields=inputs.node_inputs.candidate_fields,
            bridge_statement_fields=bridge_fields,
            nodes=nodes,
            purpose1_actor_key_manifest_fields=key_fields,
            purpose1_bridge_signature=None,
            authority=False,
        )
        replay_bridge_dag_package_v1(probe)
    except BridgeDagReplayError as exc:
        _fail(FAIL_PACKAGE_PREFLIGHT, f"unsigned full-DAG replay failed with {exc.code}: {exc.detail}")
    except BridgeDagNodeBuildError:
        raise
    except Exception as exc:
        _fail(FAIL_PACKAGE_PREFLIGHT, f"unsigned full-DAG package preflight failed: {exc}")

    return build_bridge_dag_replay_package_v1(
        purpose_id=inputs.purpose_id,
        candidate_fields=inputs.node_inputs.candidate_fields,
        bridge_statement_fields=bridge_fields,
        nodes=nodes,
        purpose1_actor_key_manifest_fields=key_fields,
        purpose1_bridge_signature=inputs.purpose1_bridge_signature,
        authority=inputs.authority,
    )


__all__ = [
    "BridgeDagNodeBuildError",
    "BridgeDagNodeBuildInputsV1",
    "BridgeDagPackageBuildInputsV1",
    "DYNAMIC_OBJECT_SCHEMAS",
    "FAIL_BASIS_BINDING",
    "FAIL_CANDIDATE_BINDING",
    "FAIL_COUNT",
    "FAIL_CROSS_ROLE",
    "FAIL_FIELD_SET",
    "FAIL_INPUT_TYPE",
    "FAIL_PACKAGE_PREFLIGHT",
    "FAIL_PREIMAGE",
    "FAIL_ROLE_LAYOUT",
    "FAIL_SIGNATURE_PHASE",
    "M3ExecutionBindingContractFieldsV1",
    "SEALED_SPLIT_FIELD_NAMES",
    "build_bridge_dag_nodes_v1",
    "build_bridge_dag_replay_package_from_inputs_v1",
]
