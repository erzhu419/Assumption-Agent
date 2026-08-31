"""Strict, target-blind Q1 start/completion control-plane wire.

This module fills only the wire gap after the immutable 14-of-20 prestart
receipt.  It does not invoke either endpoint, start Q1, read target/split/role
inputs, publish state, or consume authorization.  The existing archive actors
remain unchanged.

The final receipt deliberately does not embed its own root.  Its canonical
authority object carries an exact ``SELF_CONTENT_ROOT`` rule for output slot 8;
after strict replay the receipt content root is projected into that external
slot.  This avoids a Gate-19/receipt self-reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Final, NoReturn

from .phase3_q1_formal_archive_contract_v1 import Q1_OUTPUT_SLOT_NAMES
from .phase3_q1_prestart_wire_v1 import (
    FORMAL_RESOURCE_ENVELOPE,
    Q1_GATE_REGISTRY,
    Q1_GATE_REGISTRY_ROOT,
)
from .strict_cbor_v1 import canonical_cbor_decode, canonical_cbor_encode, content_hash


Q1_DUAL_REPLAY_AGREEMENT_TAG: Final = 0x3B10
Q1_TARGET_BLIND_ACCESS_LEDGER_TAG: Final = 0x3B11
Q1_COMPLETION_CORE_TAG: Final = 0x3B12
Q1_COMPLETION_RECEIPT_TAG: Final = 0x3B13

DUAL_REPLAY_AGREEMENT_SCHEMA_ID: Final = b"hegel-q1-dual-replay-agreement/1"
TARGET_BLIND_ACCESS_LEDGER_SCHEMA_ID: Final = (
    b"hegel-q1-target-blind-access-ledger/1"
)
COMPLETION_CORE_SCHEMA_ID: Final = b"hegel-q1-completion-core/1"
COMPLETION_RECEIPT_SCHEMA_ID: Final = b"hegel-q1-completion-receipt/1"
SIGNATURE_COMPLETION_ROW_SCHEMA_ID: Final = (
    b"hegel-q1-signature-completion-row/1"
)
ENDPOINT_COMPLETION_EVIDENCE_SCHEMA_ID: Final = (
    b"hegel-q1-endpoint-completion-evidence/1"
)
ISOLATION_POSTCHECK_ROW_SCHEMA_ID: Final = (
    b"hegel-q1-isolation-postcheck-row/1"
)
START_COMPLETION_TAG_REGISTRY_SCHEMA_ID: Final = (
    b"hegel-q1-start-completion-tag-registry/1"
)
START_GATE_SUBJECT_REGISTRY_SCHEMA_ID: Final = (
    b"hegel-q1-start-gate-subject-registry/1"
)
COMPLETE_AUTHORITY_SCHEMA_ID: Final = b"hegel-q1-complete-authority/1"

START_COMPLETION_TAG_REGISTRY_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/START_COMPLETION_TAG_REGISTRY/V1"
)
START_GATE_SUBJECT_REGISTRY_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/START_GATE_SUBJECT_REGISTRY/V1"
)
START_GATE_EVIDENCE_ROOT_DOMAIN: Final = "HEGEL/Q1/START_GATE_EVIDENCE/V1"
SIGNATURE_COMPLETION_ROW_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/SIGNATURE_COMPLETION_ROW/V1"
)
ENDPOINT_COMPLETION_EVIDENCE_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/ENDPOINT_COMPLETION_EVIDENCE/V1"
)
DUAL_REPLAY_AGREEMENT_ROOT_DOMAIN: Final = "HEGEL/Q1/DUAL_REPLAY_AGREEMENT/V1"
TARGET_BLIND_ACCESS_LEDGER_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/TARGET_BLIND_ACCESS_LEDGER/V1"
)
COMPLETION_CORE_ROOT_DOMAIN: Final = "HEGEL/Q1/COMPLETION_CORE/V1"
COMPLETION_RECEIPT_ROOT_DOMAIN: Final = "HEGEL/Q1/COMPLETION_RECEIPT/V1"
ENDPOINT_RESULT_JSON_ROOT_DOMAIN: Final = b"HEGEL/Q1/ENDPOINT_RESULT/V1\0"
HOST_RESULT_JSON_ROOT_DOMAIN: Final = b"HEGEL/Q1/HOST_RESULT/V1\0"
FORMAL_FILE_CLOSURE_BYTES_DOMAIN: Final = b"HEGEL/Q1/FORMAL_FILE_CLOSURE/V1\0"
OUTPUT_FILE_CLOSURE_BYTES_DOMAIN: Final = b"HEGEL/Q1/OUTPUT_TREE_CLOSURE/V1\0"
SOURCE_MODULE_CLOSURE_JSON_ROOT_DOMAIN: Final = (
    b"HEGEL/Q1/ACTOR_SOURCE_CLOSURE/V1\0"
)
MOUNT_MANIFEST_JSON_ROOT_DOMAIN: Final = b"HEGEL/Q1/MOUNT_REGISTRY/V1\0"
SCRATCH_CLEANUP_JSON_ROOT_DOMAIN: Final = b"HEGEL/Q1/SCRATCH_ABSENCE/V1\0"
ISOLATION_PRECHECK_ROOT_DOMAIN: Final = "HEGEL/Q1/ISOLATION_PRECHECK/V1"
ISOLATION_POSTCHECK_ROOT_DOMAIN: Final = "HEGEL/Q1/ISOLATION_POSTCHECK/V1"

_SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")

_ENDPOINT_RESULT_KEYS: Final = frozenset(
    {
        "action_id",
        "actor_id",
        "actual_formal_peak_scratch_bytes",
        "actual_formal_terminal_live_scratch_bytes",
        "capsule_digest",
        "closure_bundle_root",
        "closure_semantics_version",
        "diagnostic_only",
        "dsl_version",
        "formal_archive_materialized",
        "formal_archive_payload_bytes",
        "formal_capacity_limits_by_signature",
        "formal_output_bytes_limit",
        "formal_scratch_bytes_limit",
        "full_v16_structural_limits_applied",
        "implementation_id",
        "maximum_ast_depth",
        "maximum_ast_node_count",
        "partitions",
        "prestart_admission_receipt_root",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_gate_total",
        "q1_output_slots",
        "q1_state",
        "role_evaluation_performed",
        "run_genesis_root",
        "schema_version",
        "source_commit",
        "split_accessed",
        "status",
        "target_truth_accessed",
    }
)
_HOST_RESULT_KEYS: Final = frozenset(
    {
        "schema_version",
        "actor_id",
        "implementation_id",
        "action_id",
        "status",
        "capsule_digest",
        "source_commit",
        "run_genesis_root",
        "prestart_admission_receipt_root",
        "q1_state",
        "q1_gate_count",
        "q1_gate_total",
        "q1_gate_mask",
        "q1_output_slots",
        "formal_capacity_limits_by_signature",
        "formal_output_bytes_limit",
        "formal_scratch_bytes_limit",
        "semantic_binding_root",
        "projection_profile_root",
        "closure_bundle_root",
        "archive_payload_bytes_per_endpoint",
        "formal_file_count_per_endpoint",
        "signatures",
        "python_rust_formal_bytes_exact",
        "python_rust_replayed_roots_exact",
        "target_truth_accessed",
        "split_accessed",
        "role_evaluation_performed",
        "target_split_role_absence_attested",
        "q1_completion_receipt_issued",
        "state_published",
    }
)

START_COMPLETION_TAG_REGISTRY: Final = (
    (Q1_DUAL_REPLAY_AGREEMENT_TAG, b"Q1_DUAL_REPLAY_AGREEMENT"),
    (Q1_TARGET_BLIND_ACCESS_LEDGER_TAG, b"Q1_TARGET_BLIND_ACCESS_LEDGER"),
    (Q1_COMPLETION_CORE_TAG, b"Q1_COMPLETION_CORE"),
    (Q1_COMPLETION_RECEIPT_TAG, b"Q1_COMPLETION_RECEIPT"),
)

# The original prestart registry freezes all twenty names but exact subject
# shapes only for Gates 1..14.  This disjoint registry completes the shapes
# without changing the already-published prestart registry root.
Q1_START_GATE_SUBJECT_REGISTRY: Final = (
    (
        15,
        (
            (b"run_genesis_root", b"root32"),
            (b"prestart_admission_receipt_root", b"root32"),
            (b"ordered_endpoint_completion_evidence", b"endpoint_evidence_rows"),
        ),
    ),
    (
        16,
        (
            (b"python_endpoint_evidence_root", b"root32"),
            (b"rust_endpoint_evidence_root", b"root32"),
            (b"formal_file_closure_root", b"root32"),
            (b"ordered_shared_signature_rows", b"signature_rows"),
            (b"closure_bundle_root", b"root32"),
            (b"python_rust_formal_bytes_exact", b"true"),
            (b"state_preimages_exact", b"true"),
        ),
    ),
    (
        17,
        (
            (b"dual_replay_agreement_object", b"dual_replay_object"),
            (b"dual_replay_agreement_root", b"root32"),
        ),
    ),
    (
        18,
        (
            (b"target_blind_access_ledger_object", b"access_ledger_object"),
            (b"target_blind_access_ledger_root", b"root32"),
        ),
    ),
    (
        19,
        (
            (b"closure_bundle_root", b"root32"),
            (b"completion_core_object", b"completion_core_object"),
            (b"completion_core_root", b"root32"),
            (b"completion_seal_policy", b"completion_seal_policy"),
        ),
    ),
    (
        20,
        (
            (b"completion_core_root", b"root32"),
            (b"q1_complete_authority_object", b"complete_authority_object"),
            (b"slot8_projection_policy", b"slot8_policy"),
        ),
    ),
)

Q1_START_GATE_COUNT: Final = 6
Q1_COMPLETE_GATE_COUNT: Final = 20
Q1_COMPLETE_GATE_MASK: Final = (1 << Q1_COMPLETE_GATE_COUNT) - 1

COMPLETION_SEAL_POLICY: Final = (
    1,
    b"hegel-q1-completion-seal-policy/1",
    b"COMPLETION_CORE_THEN_GATE19_GATE20_THEN_FINAL_RECEIPT",
    b"FINAL_RECEIPT_ROOT_IS_EXTERNAL_OUTPUT_SLOT8",
)
SLOT8_SELF_CONTENT_ROOT_POLICY: Final = (
    1,
    b"hegel-q1-completion-slot8-policy/1",
    b"SELF_CONTENT_ROOT",
    Q1_OUTPUT_SLOT_NAMES[7],
    COMPLETION_RECEIPT_ROOT_DOMAIN.encode("ascii"),
)

_SIGNATURE_LIMITS: Final = {
    signature_id: dict(rows)
    for signature_id, rows in FORMAL_RESOURCE_ENVELOPE[2]
}
_ACTOR_LIMITS: Final = {
    actor_id: dict(rows)
    for _ordinal, actor_id, rows in FORMAL_RESOURCE_ENVELOPE[3]
}
_ENDPOINT_ACTORS: Final = (b"python", b"rust")
_ALL_ACTORS: Final = (b"python", b"rust", b"host")

# The predecessor Gate-11 envelope is immutable and therefore remains in
# ``_ACTOR_LIMITS`` exactly as received.  Q1-start successor evidence uses the
# explicitly authorized corrections below for endpoint scratch guard 10 and
# trusted-host memory guard 11.
FORMAL_ENDPOINT_SCRATCH_BYTES_LIMIT: Final = 134_217_728
FORMAL_HOST_MEMORY_BYTES_LIMIT: Final = 2_147_483_648


def _actor_guard_limit_v1(actor_id: bytes, guard_id: int) -> int:
    if actor_id in _ENDPOINT_ACTORS and guard_id == 10:
        return FORMAL_ENDPOINT_SCRATCH_BYTES_LIMIT
    if actor_id == b"host" and guard_id == 11:
        return FORMAL_HOST_MEMORY_BYTES_LIMIT
    return _ACTOR_LIMITS[actor_id][guard_id]

_FORMAL_CAPACITY_LIMIT_FIELDS: Final = (
    "input_signature_id",
    "maximum_ast_depth",
    "maximum_ast_node_count",
    "maximum_raw_operator_applications",
    "maximum_behavior_classes",
    "maximum_visible_frontier_points",
    "maximum_visible_frontier_points_per_class",
    "maximum_continuation_bank_points",
    "maximum_continuation_bank_points_per_class",
    "maximum_work_queue_points",
    "maximum_saturation_rounds",
    "maximum_wall_time_seconds",
)

ISOLATION_PRECHECK_RECEIPT_SCHEMA_ID: Final = (
    b"hegel-q1-isolation-precheck-receipt/1"
)
ISOLATION_POSTCHECK_RECEIPT_SCHEMA_ID: Final = (
    b"hegel-q1-isolation-postcheck-receipt/1"
)
RUNTIME_MOUNT_REGISTRY_SCHEMA_ID: Final = b"hegel-q1-runtime-mount-registry/1"
RUNTIME_MOUNT_REGISTRY_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/RUNTIME_MOUNT_REGISTRY/V1"
)
FORMAL_INPUT_KEY_ABSENCE_SCHEMA_ID: Final = (
    b"hegel-q1-formal-input-key-absence/1"
)
FORMAL_INPUT_KEY_ABSENCE_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/FORMAL_INPUT_KEY_ABSENCE/V1"
)


class Q1StartCompletionWireError(ValueError):
    """Stable rejection from the Q1 start/completion wire."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q1StartCompletionWireError(code, detail)


def _root32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        _fail("REJECT_Q1_START_ROOT", f"{label} must be exactly 32 bytes")
    return value


def _uint(value: object, label: str, maximum: int = (1 << 64) - 1) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        _fail("REJECT_Q1_START_UINT", f"{label} must be an exact bounded uint")
    return value


def _exact_true(value: object, label: str) -> None:
    if value is not True:
        _fail("REJECT_Q1_START_BOOL", f"{label} must be exact true")


def _exact_false(value: object, label: str) -> None:
    if value is not False:
        _fail("REJECT_Q1_START_BOOL", f"{label} must be exact false")


def _strict_object(payload: bytes, label: str) -> tuple[object, ...]:
    if type(payload) is not bytes:
        _fail("REJECT_Q1_START_CBOR", f"{label} payload must be bytes")
    try:
        value = canonical_cbor_decode(payload)
    except ValueError as error:
        _fail("REJECT_Q1_START_CBOR", f"{label}: {error}")
    if type(value) is not tuple or canonical_cbor_encode(value) != payload:
        _fail("REJECT_Q1_START_CBOR", f"{label} is not strict canonical array CBOR")
    return value


def _typed_tuple(value: object, length: int, label: str) -> tuple[object, ...]:
    if type(value) is not tuple or len(value) != length:
        _fail("REJECT_Q1_START_WIRE", f"{label} tuple shape differs")
    return value


def _wire_equal(value: object, expected: object) -> bool:
    """Compare decoded wire values without Python's bool/int aliasing."""

    if type(value) is not type(expected):
        return False
    if type(expected) is tuple:
        return len(value) == len(expected) and all(
            _wire_equal(item, expected_item)
            for item, expected_item in zip(value, expected, strict=True)
        )
    return value == expected


def _canonical_json_bytes_v1(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        _fail("REJECT_Q1_EVIDENCE_JSON", f"canonical JSON encoding failed: {error}")


def _strict_canonical_json_object_v1(
    payload: bytes,
    label: str,
    *,
    maximum: int,
) -> dict[str, object]:
    if type(payload) is not bytes or not payload or len(payload) > maximum:
        _fail("REJECT_Q1_EVIDENCE_JSON", f"{label} byte boundary differs")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value
        return result

    def reject_number(value: str) -> NoReturn:
        raise ValueError(f"non-integer number {value}")

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=unique_object,
            parse_float=reject_number,
            parse_constant=reject_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        _fail("REJECT_Q1_EVIDENCE_JSON", f"{label}: {error}")
    if type(value) is not dict or _canonical_json_bytes_v1(value) != payload:
        _fail(
            "REJECT_Q1_EVIDENCE_JSON",
            f"{label} must be one strict canonical JSON object",
        )
    return value


def _json_domain_root(domain: bytes, value: object) -> bytes:
    return sha256(domain + _canonical_json_bytes_v1(value)).digest()


def _tagged_json_root(value: object, label: str) -> bytes:
    if type(value) is not str or _SHA256_ID.fullmatch(value) is None:
        _fail("REJECT_Q1_EVIDENCE_JSON", f"{label} must be one sha256 root")
    return bytes.fromhex(value[7:])


def _formal_capacity_limit_rows_v1() -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    for signature_id in (1, 2):
        limits = _SIGNATURE_LIMITS[signature_id]
        result.append(
            dict(
                zip(
                    _FORMAL_CAPACITY_LIMIT_FIELDS,
                    (
                        signature_id,
                        3,
                        6,
                        *(limits[index] for index in range(1, 9)),
                        3_600,
                    ),
                    strict=True,
                )
            )
        )
    return result


def signature_guard_ceiling_rows_v1(
    input_signature_id: int,
) -> tuple[tuple[int, int], ...]:
    """Return the eight frozen guard ceilings; these are not observed peaks."""

    if type(input_signature_id) is not int or input_signature_id not in (1, 2):
        _fail("REJECT_Q1_SIGNATURE_COMPLETION", "signature ID differs")
    limits = _SIGNATURE_LIMITS[input_signature_id]
    return tuple((guard_id, limits[guard_id]) for guard_id in range(1, 9))


def _validate_success_result_common_v1(value: dict[str, object], label: str) -> None:
    fixed = {
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_total": 20,
        "q1_gate_mask": 0x3FFF,
        "q1_output_slots": [None] * 8,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
    }
    if any(not _wire_equal(value.get(name), expected) for name, expected in fixed.items()):
        _fail("REJECT_Q1_EVIDENCE_RESULT", f"{label} fixed authority boundary differs")
    for name in (
        "capsule_digest",
        "run_genesis_root",
        "prestart_admission_receipt_root",
        "closure_bundle_root",
    ):
        _tagged_json_root(value.get(name), f"{label} {name}")
    source_commit = value.get("source_commit")
    if type(source_commit) is not str or _SOURCE_COMMIT.fullmatch(source_commit) is None:
        _fail("REJECT_Q1_EVIDENCE_RESULT", f"{label} source commit differs")
    if value.get("formal_capacity_limits_by_signature") != _formal_capacity_limit_rows_v1():
        _fail("REJECT_Q1_EVIDENCE_RESULT", f"{label} frozen guard ceilings differ")


def _validate_expected_result_bindings_v1(
    value: dict[str, object],
    *,
    expected_capsule_digest: str,
    expected_source_commit: str,
    expected_run_genesis_root: bytes,
    expected_prestart_admission_receipt_root: bytes,
    expected_closure_bundle_root: bytes,
    expected_signature_rows: tuple["Q1SignatureCompletionRowV1", ...],
    signature_key: str,
    label: str,
) -> None:
    if type(expected_capsule_digest) is not str or _SHA256_ID.fullmatch(
        expected_capsule_digest
    ) is None:
        _fail("REJECT_Q1_EVIDENCE_BINDING", "expected capsule digest differs")
    if type(expected_source_commit) is not str or _SOURCE_COMMIT.fullmatch(
        expected_source_commit
    ) is None:
        _fail("REJECT_Q1_EVIDENCE_BINDING", "expected source commit differs")
    for name, root in (
        ("run genesis", expected_run_genesis_root),
        ("prestart receipt", expected_prestart_admission_receipt_root),
        ("closure bundle", expected_closure_bundle_root),
    ):
        _root32(root, f"expected {name}")
    if (
        type(expected_signature_rows) is not tuple
        or len(expected_signature_rows) != 2
        or any(type(row) is not Q1SignatureCompletionRowV1 for row in expected_signature_rows)
        or tuple(row.input_signature_id for row in expected_signature_rows) != (1, 2)
    ):
        _fail("REJECT_Q1_EVIDENCE_BINDING", "expected signature rows differ")
    expected_scalars = {
        "capsule_digest": expected_capsule_digest,
        "source_commit": expected_source_commit,
        "run_genesis_root": "sha256:" + expected_run_genesis_root.hex(),
        "prestart_admission_receipt_root": (
            "sha256:" + expected_prestart_admission_receipt_root.hex()
        ),
        "closure_bundle_root": "sha256:" + expected_closure_bundle_root.hex(),
    }
    if any(value.get(name) != expected for name, expected in expected_scalars.items()):
        _fail("REJECT_Q1_EVIDENCE_BINDING", f"{label} current run binding differs")
    signatures = value.get(signature_key)
    if type(signatures) is not list or len(signatures) != 2:
        _fail("REJECT_Q1_EVIDENCE_BINDING", f"{label} signature cardinality differs")
    for expected, actual in zip(expected_signature_rows, signatures, strict=True):
        if type(actual) is not dict or actual.get("input_signature_id") != expected.input_signature_id:
            _fail("REJECT_Q1_EVIDENCE_BINDING", f"{label} signature order differs")
        expected_roots = {
            "fixed_point_record_root": "sha256:" + expected.fixed_point_record_root.hex(),
            "signature_manifest_root": "sha256:" + expected.signature_manifest_root.hex(),
            "saturation_state_root": "sha256:" + expected.saturation_state_root.hex(),
        }
        if any(actual.get(name) != root for name, root in expected_roots.items()):
            _fail("REJECT_Q1_EVIDENCE_BINDING", f"{label} signature roots differ")


def endpoint_result_root_v1(
    actor_id: bytes,
    canonical_json_payload: bytes,
    *,
    expected_capsule_digest: str,
    expected_source_commit: str,
    expected_run_genesis_root: bytes,
    expected_prestart_admission_receipt_root: bytes,
    expected_closure_bundle_root: bytes,
    expected_signature_rows: tuple["Q1SignatureCompletionRowV1", ...],
) -> bytes:
    """Validate and bind one successful frozen endpoint-result JSON object."""

    if type(actor_id) is not bytes or actor_id not in _ENDPOINT_ACTORS:
        _fail("REJECT_Q1_EVIDENCE_RESULT", "endpoint actor differs")
    value = _strict_canonical_json_object_v1(
        canonical_json_payload,
        f"{actor_id.decode('ascii')} endpoint result",
        maximum=_ACTOR_LIMITS[actor_id][9],
    )
    if frozenset(value) != _ENDPOINT_RESULT_KEYS:
        _fail("REJECT_Q1_EVIDENCE_RESULT", "endpoint result key closure differs")
    actor_name = actor_id.decode("ascii")
    fixed = {
        "schema_version": "hegel-phase3a-q1-full-node6-formal-archive-materializer/1",
        "actor_id": actor_name,
        "implementation_id": f"hegel-{actor_name}-q1-formal-archive-materializer-v1",
        "action_id": "materialize-full-node6-formal-archive-v1",
        "status": "FULL_NODE6_FORMAL_ARCHIVE_MATERIALIZED_Q1_NOT_RUN",
        "diagnostic_only": False,
        "formal_archive_materialized": True,
        "full_v16_structural_limits_applied": True,
        "maximum_ast_depth": 3,
        "maximum_ast_node_count": 6,
        "formal_output_bytes_limit": _ACTOR_LIMITS[actor_id][9],
        "formal_scratch_bytes_limit": FORMAL_ENDPOINT_SCRATCH_BYTES_LIMIT,
        "actual_formal_terminal_live_scratch_bytes": 0,
    }
    if any(not _wire_equal(value.get(name), expected) for name, expected in fixed.items()):
        _fail("REJECT_Q1_EVIDENCE_RESULT", "endpoint success boundary differs")
    _validate_success_result_common_v1(value, "endpoint result")
    peak_scratch = value.get("actual_formal_peak_scratch_bytes")
    payload_bytes = value.get("formal_archive_payload_bytes")
    if (
        type(peak_scratch) is not int
        or peak_scratch < 0
        or peak_scratch > FORMAL_ENDPOINT_SCRATCH_BYTES_LIMIT
        or type(payload_bytes) is not int
        or payload_bytes <= 0
        or payload_bytes > _ACTOR_LIMITS[actor_id][9]
    ):
        _fail("REJECT_Q1_EVIDENCE_RESULT", "endpoint resource evidence differs")
    partitions = value.get("partitions")
    if type(partitions) is not list or len(partitions) != 2:
        _fail("REJECT_Q1_EVIDENCE_RESULT", "endpoint signature cardinality differs")
    for signature_id, partition in enumerate(partitions, 1):
        if type(partition) is not dict or partition.get("input_signature_id") != signature_id:
            _fail("REJECT_Q1_EVIDENCE_RESULT", "endpoint signature order differs")
        for name in (
            "fixed_point_record_root",
            "signature_manifest_root",
            "saturation_state_root",
        ):
            _tagged_json_root(partition.get(name), f"signature {signature_id} {name}")
    _validate_expected_result_bindings_v1(
        value,
        expected_capsule_digest=expected_capsule_digest,
        expected_source_commit=expected_source_commit,
        expected_run_genesis_root=expected_run_genesis_root,
        expected_prestart_admission_receipt_root=(
            expected_prestart_admission_receipt_root
        ),
        expected_closure_bundle_root=expected_closure_bundle_root,
        expected_signature_rows=expected_signature_rows,
        signature_key="partitions",
        label="endpoint result",
    )
    return sha256(ENDPOINT_RESULT_JSON_ROOT_DOMAIN + canonical_json_payload).digest()


def host_replay_result_root_v1(
    canonical_json_payload: bytes,
    *,
    expected_capsule_digest: str,
    expected_source_commit: str,
    expected_run_genesis_root: bytes,
    expected_prestart_admission_receipt_root: bytes,
    expected_closure_bundle_root: bytes,
    expected_signature_rows: tuple["Q1SignatureCompletionRowV1", ...],
) -> bytes:
    """Validate and bind the successful trusted-host replay JSON object."""

    value = _strict_canonical_json_object_v1(
        canonical_json_payload,
        "host replay result",
        maximum=_ACTOR_LIMITS[b"host"][9],
    )
    if frozenset(value) != _HOST_RESULT_KEYS:
        _fail("REJECT_Q1_EVIDENCE_RESULT", "host result key closure differs")
    fixed = {
        "schema_version": "hegel-q1-formal-host-replay-result/1",
        "actor_id": "host",
        "implementation_id": "hegel-host-q1-formal-archive-replay-v1",
        "action_id": "replay-full-node6-formal-archive-v1",
        "status": "DUAL_FORMAL_ARCHIVE_EXACT_AGREEMENT",
        "formal_output_bytes_limit": _ACTOR_LIMITS[b"python"][9],
        "formal_scratch_bytes_limit": FORMAL_ENDPOINT_SCRATCH_BYTES_LIMIT,
        "python_rust_formal_bytes_exact": True,
        "python_rust_replayed_roots_exact": True,
        "target_split_role_absence_attested": True,
        "q1_completion_receipt_issued": False,
        "state_published": False,
    }
    if any(not _wire_equal(value.get(name), expected) for name, expected in fixed.items()):
        _fail("REJECT_Q1_EVIDENCE_RESULT", "host success boundary differs")
    _validate_success_result_common_v1(value, "host result")
    for name in ("semantic_binding_root", "projection_profile_root"):
        _tagged_json_root(value.get(name), f"host result {name}")
    signatures = value.get("signatures")
    if type(signatures) is not list or len(signatures) != 2:
        _fail("REJECT_Q1_EVIDENCE_RESULT", "host signature cardinality differs")
    for signature_id, signature in enumerate(signatures, 1):
        if type(signature) is not dict or signature.get("input_signature_id") != signature_id:
            _fail("REJECT_Q1_EVIDENCE_RESULT", "host signature order differs")
        for name in (
            "universe_root",
            "fixed_point_record_root",
            "signature_manifest_root",
            "saturation_state_root",
        ):
            _tagged_json_root(signature.get(name), f"host signature {signature_id} {name}")
    _validate_expected_result_bindings_v1(
        value,
        expected_capsule_digest=expected_capsule_digest,
        expected_source_commit=expected_source_commit,
        expected_run_genesis_root=expected_run_genesis_root,
        expected_prestart_admission_receipt_root=(
            expected_prestart_admission_receipt_root
        ),
        expected_closure_bundle_root=expected_closure_bundle_root,
        expected_signature_rows=expected_signature_rows,
        signature_key="signatures",
        label="host result",
    )
    return sha256(HOST_RESULT_JSON_ROOT_DOMAIN + canonical_json_payload).digest()


@dataclass(frozen=True, slots=True)
class Q1FileEvidenceV1:
    relative_path: str
    payload: bytes

    def __post_init__(self) -> None:
        if (
            type(self.relative_path) is not str
            or not self.relative_path
            or self.relative_path.startswith("/")
            or any(part in ("", ".", "..") for part in self.relative_path.split("/"))
        ):
            _fail("REJECT_Q1_FILE_EVIDENCE", "relative file path differs")
        try:
            self.relative_path.encode("utf-8")
        except UnicodeEncodeError as error:
            _fail("REJECT_Q1_FILE_EVIDENCE", f"file path encoding differs: {error}")
        if type(self.payload) is not bytes:
            _fail("REJECT_Q1_FILE_EVIDENCE", "file payload must be bytes")


def _ordered_file_closure_root_v1(
    files: tuple[Q1FileEvidenceV1, ...],
    domain: bytes,
) -> bytes:
    if type(files) is not tuple or any(type(row) is not Q1FileEvidenceV1 for row in files):
        _fail("REJECT_Q1_FILE_EVIDENCE", "exact file-evidence tuple required")
    paths = tuple(row.relative_path for row in files)
    if not paths or paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        _fail("REJECT_Q1_FILE_EVIDENCE", "file path order/cardinality differs")
    digest = sha256(domain)
    for row in files:
        encoded = row.relative_path.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(row.payload).to_bytes(8, "big"))
        digest.update(row.payload)
    return digest.digest()


def formal_output_file_closure_root_v1(
    files: tuple[Q1FileEvidenceV1, ...],
) -> bytes:
    return _ordered_file_closure_root_v1(files, FORMAL_FILE_CLOSURE_BYTES_DOMAIN)


def output_file_closure_root_v1(files: tuple[Q1FileEvidenceV1, ...]) -> bytes:
    return _ordered_file_closure_root_v1(files, OUTPUT_FILE_CLOSURE_BYTES_DOMAIN)


@dataclass(frozen=True, slots=True)
class Q1SourceManifestEvidenceRowV1:
    relative_path: str
    byte_count: int
    raw_sha256: bytes

    def __post_init__(self) -> None:
        Q1FileEvidenceV1(self.relative_path, b"")
        _uint(self.byte_count, "source-manifest byte count")
        _root32(self.raw_sha256, "source-manifest raw sha256")

    def json_object(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "byte_count": self.byte_count,
            "raw_sha256": "sha256:" + self.raw_sha256.hex(),
        }


def source_module_closure_root_v1(
    rows: tuple[Q1SourceManifestEvidenceRowV1, ...],
) -> bytes:
    if type(rows) is not tuple or not rows or any(
        type(row) is not Q1SourceManifestEvidenceRowV1 for row in rows
    ):
        _fail("REJECT_Q1_SOURCE_EVIDENCE", "exact nonempty source rows required")
    paths = tuple(row.relative_path for row in rows)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        _fail("REJECT_Q1_SOURCE_EVIDENCE", "source row order/cardinality differs")
    return _json_domain_root(
        SOURCE_MODULE_CLOSURE_JSON_ROOT_DOMAIN,
        [row.json_object() for row in rows],
    )


@dataclass(frozen=True, slots=True)
class Q1MountManifestEvidenceRowV1:
    destination: str
    source: str
    read_write: bool

    def __post_init__(self) -> None:
        if (
            type(self.destination) is not str
            or not self.destination.startswith("/")
            or type(self.source) is not str
            or not self.source.startswith("/")
            or type(self.read_write) is not bool
        ):
            _fail("REJECT_Q1_MOUNT_EVIDENCE", "mount row differs")

    def json_object(self) -> dict[str, object]:
        return {
            "destination": self.destination,
            "read_write": self.read_write,
            "source": self.source,
        }


def mount_manifest_root_v1(
    rows: tuple[Q1MountManifestEvidenceRowV1, ...],
) -> bytes:
    if type(rows) is not tuple or not rows or any(
        type(row) is not Q1MountManifestEvidenceRowV1 for row in rows
    ):
        _fail("REJECT_Q1_MOUNT_EVIDENCE", "exact nonempty mount rows required")
    destinations = tuple(row.destination for row in rows)
    if destinations != tuple(sorted(destinations)) or len(destinations) != len(
        set(destinations)
    ):
        _fail("REJECT_Q1_MOUNT_EVIDENCE", "mount order/cardinality differs")
    return _json_domain_root(
        MOUNT_MANIFEST_JSON_ROOT_DOMAIN,
        [row.json_object() for row in rows],
    )


def scratch_cleanup_root_v1(scratch_path: str, *, absent: bool) -> bytes:
    if type(scratch_path) is not str or not scratch_path.startswith("/"):
        _fail("REJECT_Q1_SCRATCH_EVIDENCE", "scratch path differs")
    if absent is not True:
        _fail("REJECT_Q1_SCRATCH_EVIDENCE", "scratch absence must be exact true")
    return _json_domain_root(
        SCRATCH_CLEANUP_JSON_ROOT_DOMAIN,
        {"path": scratch_path, "absent": True},
    )


def _isolation_precheck_object_v1(
    actor_id: bytes,
    actor_implementation_root: bytes,
    source_module_closure_root: bytes,
    mount_manifest_root: bytes,
) -> tuple[object, ...]:
    if type(actor_id) is not bytes or actor_id not in _ALL_ACTORS:
        _fail("REJECT_Q1_ISOLATION_PRECHECK", "actor ID differs")
    for label, root in (
        ("actor implementation", actor_implementation_root),
        ("source module closure", source_module_closure_root),
        ("mount manifest", mount_manifest_root),
    ):
        _root32(root, label)
    return (
        1,
        ISOLATION_PRECHECK_RECEIPT_SCHEMA_ID,
        actor_id,
        actor_implementation_root,
        source_module_closure_root,
        mount_manifest_root,
        False,
        False,
        False,
        False,
    )


def isolation_precheck_root_v1(
    actor_id: bytes,
    actor_implementation_root: bytes,
    source_module_closure_root: bytes,
    mount_manifest_root: bytes,
) -> bytes:
    """Derive one actor's exact offline-isolation precheck root."""

    return content_hash(
        ISOLATION_PRECHECK_ROOT_DOMAIN,
        _isolation_precheck_object_v1(
            actor_id,
            actor_implementation_root,
            source_module_closure_root,
            mount_manifest_root,
        ),
    )


def _isolation_postcheck_object_v1(
    actor_id: bytes,
    actor_implementation_root: bytes,
    precheck_root: bytes,
    source_module_closure_root: bytes,
    mount_manifest_root: bytes,
    output_closure_root: bytes,
    scratch_cleanup_root: bytes,
) -> tuple[object, ...]:
    expected_precheck = isolation_precheck_root_v1(
        actor_id,
        actor_implementation_root,
        source_module_closure_root,
        mount_manifest_root,
    )
    if _root32(precheck_root, "isolation precheck") != expected_precheck:
        _fail("REJECT_Q1_ISOLATION_POSTCHECK", "precheck derivation differs")
    for label, root in (
        ("output closure", output_closure_root),
        ("scratch cleanup", scratch_cleanup_root),
    ):
        _root32(root, label)
    return (
        1,
        ISOLATION_POSTCHECK_RECEIPT_SCHEMA_ID,
        actor_id,
        actor_implementation_root,
        precheck_root,
        source_module_closure_root,
        mount_manifest_root,
        output_closure_root,
        scratch_cleanup_root,
        True,
        False,
        False,
        False,
        False,
    )


def isolation_postcheck_root_v1(
    actor_id: bytes,
    actor_implementation_root: bytes,
    precheck_root: bytes,
    source_module_closure_root: bytes,
    mount_manifest_root: bytes,
    output_closure_root: bytes,
    scratch_cleanup_root: bytes,
) -> bytes:
    """Derive one actor's exact postcheck root from the persisted row fields."""

    return content_hash(
        ISOLATION_POSTCHECK_ROOT_DOMAIN,
        _isolation_postcheck_object_v1(
            actor_id,
            actor_implementation_root,
            precheck_root,
            source_module_closure_root,
            mount_manifest_root,
            output_closure_root,
            scratch_cleanup_root,
        ),
    )


@dataclass(frozen=True, slots=True)
class Q1IsolationEvidenceAdapterV1:
    actor_id: bytes
    actor_implementation_root: bytes
    source_manifest_rows: tuple[Q1SourceManifestEvidenceRowV1, ...]
    mount_manifest_rows: tuple[Q1MountManifestEvidenceRowV1, ...]
    output_files: tuple[Q1FileEvidenceV1, ...]
    scratch_path: str

    def __post_init__(self) -> None:
        if type(self.actor_id) is not bytes or self.actor_id not in _ALL_ACTORS:
            _fail("REJECT_Q1_ISOLATION_EVIDENCE", "actor ID differs")
        _root32(self.actor_implementation_root, "actor implementation")
        # Force all preimages through their exact validators at construction.
        self.source_module_closure_root
        self.mount_manifest_root
        self.output_closure_root
        self.scratch_cleanup_root

    @property
    def source_module_closure_root(self) -> bytes:
        return source_module_closure_root_v1(self.source_manifest_rows)

    @property
    def mount_manifest_root(self) -> bytes:
        return mount_manifest_root_v1(self.mount_manifest_rows)

    @property
    def output_closure_root(self) -> bytes:
        return output_file_closure_root_v1(self.output_files)

    @property
    def scratch_cleanup_root(self) -> bytes:
        return scratch_cleanup_root_v1(self.scratch_path, absent=True)

    def precheck_object(self) -> tuple[object, ...]:
        return _isolation_precheck_object_v1(
            self.actor_id,
            self.actor_implementation_root,
            self.source_module_closure_root,
            self.mount_manifest_root,
        )

    @property
    def precheck_root(self) -> bytes:
        return isolation_precheck_root_v1(
            self.actor_id,
            self.actor_implementation_root,
            self.source_module_closure_root,
            self.mount_manifest_root,
        )

    def postcheck_object(self) -> tuple[object, ...]:
        return _isolation_postcheck_object_v1(
            self.actor_id,
            self.actor_implementation_root,
            self.precheck_root,
            self.source_module_closure_root,
            self.mount_manifest_root,
            self.output_closure_root,
            self.scratch_cleanup_root,
        )

    @property
    def postcheck_root(self) -> bytes:
        return isolation_postcheck_root_v1(
            self.actor_id,
            self.actor_implementation_root,
            self.precheck_root,
            self.source_module_closure_root,
            self.mount_manifest_root,
            self.output_closure_root,
            self.scratch_cleanup_root,
        )

    def isolation_row(self) -> "Q1IsolationPostcheckRowV1":
        return Q1IsolationPostcheckRowV1(
            actor_id=self.actor_id,
            actor_implementation_root=self.actor_implementation_root,
            precheck_root=self.precheck_root,
            postcheck_root=self.postcheck_root,
            source_module_closure_root=self.source_module_closure_root,
            mount_manifest_root=self.mount_manifest_root,
            output_closure_root=self.output_closure_root,
            scratch_cleanup_root=self.scratch_cleanup_root,
        )


START_COMPLETION_TAG_REGISTRY_ROOT: Final = content_hash(
    START_COMPLETION_TAG_REGISTRY_ROOT_DOMAIN,
    (
        1,
        START_COMPLETION_TAG_REGISTRY_SCHEMA_ID,
        START_COMPLETION_TAG_REGISTRY,
        Q1_GATE_REGISTRY_ROOT,
    ),
)
Q1_START_GATE_SUBJECT_REGISTRY_ROOT: Final = content_hash(
    START_GATE_SUBJECT_REGISTRY_ROOT_DOMAIN,
    (
        1,
        START_GATE_SUBJECT_REGISTRY_SCHEMA_ID,
        Q1_GATE_REGISTRY_ROOT,
        Q1_START_GATE_SUBJECT_REGISTRY,
    ),
)


@dataclass(frozen=True, slots=True)
class Q1SignatureCompletionRowV1:
    input_signature_id: int
    fixed_point_record_root: bytes
    signature_manifest_root: bytes
    saturation_state_root: bytes
    count_guard_ceiling_rows: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if type(self.input_signature_id) is not int or self.input_signature_id not in (1, 2):
            _fail("REJECT_Q1_SIGNATURE_COMPLETION", "signature ID differs")
        for label, root in (
            ("fixed point", self.fixed_point_record_root),
            ("signature manifest", self.signature_manifest_root),
            ("saturation state", self.saturation_state_root),
        ):
            _root32(root, label)
        expected = signature_guard_ceiling_rows_v1(self.input_signature_id)
        if not _wire_equal(self.count_guard_ceiling_rows, expected):
            _fail(
                "REJECT_Q1_SIGNATURE_COMPLETION",
                "count guard ceiling rows differ from the frozen envelope",
            )

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            SIGNATURE_COMPLETION_ROW_SCHEMA_ID,
            self.input_signature_id,
            self.fixed_point_record_root,
            self.signature_manifest_root,
            self.saturation_state_root,
            self.count_guard_ceiling_rows,
            True,
            True,
            True,
            0,
            0,
            0,
            0,
        )

    @property
    def row_root(self) -> bytes:
        return content_hash(SIGNATURE_COMPLETION_ROW_ROOT_DOMAIN, self.canonical_object())


def _decode_signature_row_object(value: object) -> Q1SignatureCompletionRowV1:
    row = _typed_tuple(value, 14, "signature completion row")
    if not _wire_equal(row[:2], (1, SIGNATURE_COMPLETION_ROW_SCHEMA_ID)) or not _wire_equal(
        row[7:], (True, True, True, 0, 0, 0, 0)
    ):
        _fail("REJECT_Q1_SIGNATURE_COMPLETION", "terminal fixed-point fields differ")
    result = Q1SignatureCompletionRowV1(row[2], row[3], row[4], row[5], row[6])
    if not _wire_equal(result.canonical_object(), row):
        _fail("REJECT_Q1_SIGNATURE_COMPLETION", "signature row replay differs")
    return result


def _decode_signature_rows(value: object) -> tuple[Q1SignatureCompletionRowV1, ...]:
    rows = _typed_tuple(value, 2, "ordered signature rows")
    decoded = tuple(_decode_signature_row_object(row) for row in rows)
    if tuple(row.input_signature_id for row in decoded) != (1, 2):
        _fail("REJECT_Q1_SIGNATURE_COMPLETION", "signature order differs")
    return decoded


@dataclass(frozen=True, slots=True)
class Q1EndpointCompletionEvidenceV1:
    actor_id: bytes
    actor_implementation_root: bytes
    run_genesis_root: bytes
    prestart_admission_receipt_root: bytes
    endpoint_result_root: bytes
    closure_bundle_root: bytes
    signature_rows: tuple[Q1SignatureCompletionRowV1, ...]
    actual_output_bytes: int
    actual_peak_scratch_bytes: int
    actual_peak_rss_bytes: int
    actual_wall_time_ns: int
    terminal_live_scratch_bytes: int = 0

    def __post_init__(self) -> None:
        if type(self.actor_id) is not bytes or self.actor_id not in _ENDPOINT_ACTORS:
            _fail("REJECT_Q1_ENDPOINT_COMPLETION", "endpoint actor differs")
        for label, root in (
            ("actor implementation", self.actor_implementation_root),
            ("run genesis", self.run_genesis_root),
            ("prestart receipt", self.prestart_admission_receipt_root),
            ("endpoint result", self.endpoint_result_root),
            ("closure bundle", self.closure_bundle_root),
        ):
            _root32(root, label)
        if type(self.signature_rows) is not tuple or tuple(
            type(row) for row in self.signature_rows
        ) != (Q1SignatureCompletionRowV1, Q1SignatureCompletionRowV1):
            _fail("REJECT_Q1_ENDPOINT_COMPLETION", "two exact signature rows required")
        if tuple(row.input_signature_id for row in self.signature_rows) != (1, 2):
            _fail("REJECT_Q1_ENDPOINT_COMPLETION", "signature order differs")
        measured = (
            (9, self.actual_output_bytes),
            (10, self.actual_peak_scratch_bytes),
            (11, self.actual_peak_rss_bytes),
            (12, self.actual_wall_time_ns),
        )
        for guard_id, value in measured:
            checked = _uint(value, f"endpoint guard {guard_id}")
            ceiling = _actor_guard_limit_v1(self.actor_id, guard_id)
            if guard_id == 12:
                ceiling *= 1_000_000_000
            if checked > ceiling:
                _fail("REJECT_Q1_ENDPOINT_COMPLETION", "endpoint resource guard was exceeded")
        if type(self.terminal_live_scratch_bytes) is not int or self.terminal_live_scratch_bytes != 0:
            _fail("REJECT_Q1_ENDPOINT_COMPLETION", "terminal scratch must be exactly empty")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            ENDPOINT_COMPLETION_EVIDENCE_SCHEMA_ID,
            self.actor_id,
            self.actor_implementation_root,
            self.run_genesis_root,
            self.prestart_admission_receipt_root,
            self.endpoint_result_root,
            self.closure_bundle_root,
            tuple(row.canonical_object() for row in self.signature_rows),
            self.actual_output_bytes,
            self.actual_peak_scratch_bytes,
            self.actual_peak_rss_bytes,
            self.actual_wall_time_ns,
            self.terminal_live_scratch_bytes,
            tuple((guard_id, None) for guard_id in range(1, 13)),
            b"SATURATED_NO_GUARD_HIT",
        )

    @property
    def evidence_root(self) -> bytes:
        return content_hash(
            ENDPOINT_COMPLETION_EVIDENCE_ROOT_DOMAIN,
            self.canonical_object(),
        )


def _decode_endpoint_object(value: object) -> Q1EndpointCompletionEvidenceV1:
    row = _typed_tuple(value, 16, "endpoint completion evidence")
    if (
        not _wire_equal(row[:2], (1, ENDPOINT_COMPLETION_EVIDENCE_SCHEMA_ID))
        or not _wire_equal(
            row[14], tuple((guard_id, None) for guard_id in range(1, 13))
        )
        or row[15] != b"SATURATED_NO_GUARD_HIT"
    ):
        _fail("REJECT_Q1_ENDPOINT_COMPLETION", "endpoint terminal wire differs")
    result = Q1EndpointCompletionEvidenceV1(
        actor_id=row[2],
        actor_implementation_root=row[3],
        run_genesis_root=row[4],
        prestart_admission_receipt_root=row[5],
        endpoint_result_root=row[6],
        closure_bundle_root=row[7],
        signature_rows=_decode_signature_rows(row[8]),
        actual_output_bytes=row[9],
        actual_peak_scratch_bytes=row[10],
        actual_peak_rss_bytes=row[11],
        actual_wall_time_ns=row[12],
        terminal_live_scratch_bytes=row[13],
    )
    if not _wire_equal(result.canonical_object(), row):
        _fail("REJECT_Q1_ENDPOINT_COMPLETION", "endpoint evidence replay differs")
    return result


@dataclass(frozen=True, slots=True)
class Q1DualReplayAgreementV1:
    run_genesis_root: bytes
    prestart_admission_receipt_root: bytes
    python_endpoint_evidence_root: bytes
    rust_endpoint_evidence_root: bytes
    formal_file_closure_root: bytes
    signature_rows: tuple[Q1SignatureCompletionRowV1, ...]
    closure_bundle_root: bytes
    host_actor_implementation_root: bytes
    host_replay_result_root: bytes
    actual_host_output_bytes: int
    actual_host_peak_scratch_bytes: int
    actual_host_peak_rss_bytes: int
    actual_host_wall_time_ns: int

    def __post_init__(self) -> None:
        for label, root in (
            ("run genesis", self.run_genesis_root),
            ("prestart receipt", self.prestart_admission_receipt_root),
            ("Python endpoint evidence", self.python_endpoint_evidence_root),
            ("Rust endpoint evidence", self.rust_endpoint_evidence_root),
            ("formal file closure", self.formal_file_closure_root),
            ("closure bundle", self.closure_bundle_root),
            ("host implementation", self.host_actor_implementation_root),
            ("host replay result", self.host_replay_result_root),
        ):
            _root32(root, label)
        if type(self.signature_rows) is not tuple or tuple(
            type(row) for row in self.signature_rows
        ) != (Q1SignatureCompletionRowV1, Q1SignatureCompletionRowV1):
            _fail("REJECT_Q1_DUAL_REPLAY", "two exact signature rows required")
        if tuple(row.input_signature_id for row in self.signature_rows) != (1, 2):
            _fail("REJECT_Q1_DUAL_REPLAY", "signature order differs")
        measured = (
            (9, self.actual_host_output_bytes),
            (10, self.actual_host_peak_scratch_bytes),
            (11, self.actual_host_peak_rss_bytes),
            (12, self.actual_host_wall_time_ns),
        )
        for guard_id, value in measured:
            checked = _uint(value, f"host guard {guard_id}")
            ceiling = _actor_guard_limit_v1(b"host", guard_id)
            if guard_id == 12:
                ceiling *= 1_000_000_000
            if checked > ceiling:
                _fail("REJECT_Q1_DUAL_REPLAY", "host resource guard was exceeded")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q1_DUAL_REPLAY_AGREEMENT_TAG,
            DUAL_REPLAY_AGREEMENT_SCHEMA_ID,
            START_COMPLETION_TAG_REGISTRY_ROOT,
            self.run_genesis_root,
            self.prestart_admission_receipt_root,
            self.python_endpoint_evidence_root,
            self.rust_endpoint_evidence_root,
            self.formal_file_closure_root,
            tuple(row.canonical_object() for row in self.signature_rows),
            self.closure_bundle_root,
            self.host_actor_implementation_root,
            self.host_replay_result_root,
            self.actual_host_output_bytes,
            self.actual_host_peak_scratch_bytes,
            self.actual_host_peak_rss_bytes,
            self.actual_host_wall_time_ns,
            tuple((guard_id, None) for guard_id in range(9, 13)),
            True,
            True,
            True,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def agreement_root(self) -> bytes:
        return content_hash(DUAL_REPLAY_AGREEMENT_ROOT_DOMAIN, self.canonical_object())


def _decode_dual_replay_object(value: object) -> Q1DualReplayAgreementV1:
    row = _typed_tuple(value, 21, "dual replay agreement")
    if (
        not _wire_equal(
            row[:4],
            (
                1,
                Q1_DUAL_REPLAY_AGREEMENT_TAG,
                DUAL_REPLAY_AGREEMENT_SCHEMA_ID,
                START_COMPLETION_TAG_REGISTRY_ROOT,
            ),
        )
        or not _wire_equal(
            row[17], tuple((guard_id, None) for guard_id in range(9, 13))
        )
        or not _wire_equal(row[18:], (True, True, True))
    ):
        _fail("REJECT_Q1_DUAL_REPLAY", "agreement fixed wire differs")
    result = Q1DualReplayAgreementV1(
        run_genesis_root=row[4],
        prestart_admission_receipt_root=row[5],
        python_endpoint_evidence_root=row[6],
        rust_endpoint_evidence_root=row[7],
        formal_file_closure_root=row[8],
        signature_rows=_decode_signature_rows(row[9]),
        closure_bundle_root=row[10],
        host_actor_implementation_root=row[11],
        host_replay_result_root=row[12],
        actual_host_output_bytes=row[13],
        actual_host_peak_scratch_bytes=row[14],
        actual_host_peak_rss_bytes=row[15],
        actual_host_wall_time_ns=row[16],
    )
    if not _wire_equal(result.canonical_object(), row):
        _fail("REJECT_Q1_DUAL_REPLAY", "agreement replay differs")
    return result


def decode_dual_replay_agreement_v1(payload: bytes) -> Q1DualReplayAgreementV1:
    return _decode_dual_replay_object(_strict_object(payload, "dual replay agreement"))


@dataclass(frozen=True, slots=True)
class Q1IsolationPostcheckRowV1:
    actor_id: bytes
    actor_implementation_root: bytes
    precheck_root: bytes
    postcheck_root: bytes
    source_module_closure_root: bytes
    mount_manifest_root: bytes
    output_closure_root: bytes
    scratch_cleanup_root: bytes

    def __post_init__(self) -> None:
        if type(self.actor_id) is not bytes or self.actor_id not in _ALL_ACTORS:
            _fail("REJECT_Q1_ISOLATION_POSTCHECK", "actor ID differs")
        for label, root in (
            ("actor implementation", self.actor_implementation_root),
            ("precheck", self.precheck_root),
            ("postcheck", self.postcheck_root),
            ("source module closure", self.source_module_closure_root),
            ("mount manifest", self.mount_manifest_root),
            ("output closure", self.output_closure_root),
            ("scratch cleanup", self.scratch_cleanup_root),
        ):
            _root32(root, label)
        expected_precheck = isolation_precheck_root_v1(
            self.actor_id,
            self.actor_implementation_root,
            self.source_module_closure_root,
            self.mount_manifest_root,
        )
        if self.precheck_root != expected_precheck:
            _fail("REJECT_Q1_ISOLATION_POSTCHECK", "precheck derivation differs")
        expected_postcheck = isolation_postcheck_root_v1(
            self.actor_id,
            self.actor_implementation_root,
            self.precheck_root,
            self.source_module_closure_root,
            self.mount_manifest_root,
            self.output_closure_root,
            self.scratch_cleanup_root,
        )
        if self.postcheck_root != expected_postcheck:
            _fail("REJECT_Q1_ISOLATION_POSTCHECK", "postcheck derivation differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            ISOLATION_POSTCHECK_ROW_SCHEMA_ID,
            self.actor_id,
            self.actor_implementation_root,
            self.precheck_root,
            self.postcheck_root,
            self.source_module_closure_root,
            self.mount_manifest_root,
            self.output_closure_root,
            self.scratch_cleanup_root,
            True,
            False,
            False,
            False,
            False,
        )


def _decode_isolation_row(value: object) -> Q1IsolationPostcheckRowV1:
    row = _typed_tuple(value, 15, "isolation postcheck row")
    if not _wire_equal(row[:2], (1, ISOLATION_POSTCHECK_ROW_SCHEMA_ID)) or not _wire_equal(
        row[10:], (True, False, False, False, False)
    ):
        _fail("REJECT_Q1_ISOLATION_POSTCHECK", "postcheck flags differ")
    result = Q1IsolationPostcheckRowV1(*row[2:10])
    if not _wire_equal(result.canonical_object(), row):
        _fail("REJECT_Q1_ISOLATION_POSTCHECK", "postcheck replay differs")
    return result


def runtime_mount_registry_root_v1(
    actor_rows: tuple[Q1IsolationPostcheckRowV1, ...],
) -> bytes:
    """Aggregate the exact three actor mount-manifest roots in runtime order."""

    if type(actor_rows) is not tuple or tuple(type(row) for row in actor_rows) != (
        Q1IsolationPostcheckRowV1,
    ) * 3:
        _fail("REJECT_Q1_RUNTIME_MOUNT_REGISTRY", "three exact actor rows required")
    if tuple(row.actor_id for row in actor_rows) != _ALL_ACTORS:
        _fail("REJECT_Q1_RUNTIME_MOUNT_REGISTRY", "actor order differs")
    return content_hash(
        RUNTIME_MOUNT_REGISTRY_ROOT_DOMAIN,
        (
            1,
            RUNTIME_MOUNT_REGISTRY_SCHEMA_ID,
            tuple((row.actor_id, row.mount_manifest_root) for row in actor_rows),
            False,
            False,
            False,
            False,
        ),
    )


def formal_input_key_registry_absence_root_v1() -> bytes:
    """Return the exact frozen prestart formal-input-key absence root."""

    return content_hash(
        FORMAL_INPUT_KEY_ABSENCE_ROOT_DOMAIN,
        (
            1,
            FORMAL_INPUT_KEY_ABSENCE_SCHEMA_ID,
            (),
            False,
            False,
            False,
        ),
    )


@dataclass(frozen=True, slots=True)
class Q1TargetBlindAccessLedgerV1:
    run_genesis_root: bytes
    prestart_admission_receipt_root: bytes
    dual_replay_agreement_root: bytes
    mount_manifest_root: bytes
    input_key_registry_absence_root: bytes
    actor_rows: tuple[Q1IsolationPostcheckRowV1, ...]

    def __post_init__(self) -> None:
        for label, root in (
            ("run genesis", self.run_genesis_root),
            ("prestart receipt", self.prestart_admission_receipt_root),
            ("dual replay agreement", self.dual_replay_agreement_root),
            ("mount manifest", self.mount_manifest_root),
            ("input-key absence", self.input_key_registry_absence_root),
        ):
            _root32(root, label)
        if type(self.actor_rows) is not tuple or tuple(
            type(row) for row in self.actor_rows
        ) != (Q1IsolationPostcheckRowV1,) * 3:
            _fail("REJECT_Q1_ACCESS_LEDGER", "three exact actor postchecks required")
        if tuple(row.actor_id for row in self.actor_rows) != _ALL_ACTORS:
            _fail("REJECT_Q1_ACCESS_LEDGER", "actor postcheck order differs")
        if self.mount_manifest_root != runtime_mount_registry_root_v1(
            self.actor_rows
        ):
            _fail("REJECT_Q1_ACCESS_LEDGER", "runtime mount registry differs")
        if (
            self.input_key_registry_absence_root
            != formal_input_key_registry_absence_root_v1()
        ):
            _fail("REJECT_Q1_ACCESS_LEDGER", "formal input-key absence differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q1_TARGET_BLIND_ACCESS_LEDGER_TAG,
            TARGET_BLIND_ACCESS_LEDGER_SCHEMA_ID,
            START_COMPLETION_TAG_REGISTRY_ROOT,
            self.run_genesis_root,
            self.prestart_admission_receipt_root,
            self.dual_replay_agreement_root,
            self.mount_manifest_root,
            self.input_key_registry_absence_root,
            tuple(row.canonical_object() for row in self.actor_rows),
            (),
            False,
            False,
            False,
            False,
            True,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def ledger_root(self) -> bytes:
        return content_hash(
            TARGET_BLIND_ACCESS_LEDGER_ROOT_DOMAIN,
            self.canonical_object(),
        )


def _decode_access_ledger_object(value: object) -> Q1TargetBlindAccessLedgerV1:
    row = _typed_tuple(value, 16, "target-blind access ledger")
    if (
        not _wire_equal(
            row[:4],
            (
                1,
                Q1_TARGET_BLIND_ACCESS_LEDGER_TAG,
                TARGET_BLIND_ACCESS_LEDGER_SCHEMA_ID,
                START_COMPLETION_TAG_REGISTRY_ROOT,
            ),
        )
        or row[10] != ()
        or not _wire_equal(row[11:], (False, False, False, False, True))
    ):
        _fail("REJECT_Q1_ACCESS_LEDGER", "access-ledger fixed wire differs")
    actor_objects = _typed_tuple(row[9], 3, "access-ledger actor rows")
    result = Q1TargetBlindAccessLedgerV1(
        run_genesis_root=row[4],
        prestart_admission_receipt_root=row[5],
        dual_replay_agreement_root=row[6],
        mount_manifest_root=row[7],
        input_key_registry_absence_root=row[8],
        actor_rows=tuple(_decode_isolation_row(value) for value in actor_objects),
    )
    if not _wire_equal(result.canonical_object(), row):
        _fail("REJECT_Q1_ACCESS_LEDGER", "access-ledger replay differs")
    return result


def decode_target_blind_access_ledger_v1(
    payload: bytes,
) -> Q1TargetBlindAccessLedgerV1:
    return _decode_access_ledger_object(_strict_object(payload, "access ledger"))


def _gate15_subjects(
    subjects: object,
) -> tuple[bytes, bytes, tuple[Q1EndpointCompletionEvidenceV1, ...]]:
    row = _typed_tuple(subjects, 3, "Gate 15 subjects")
    run_root = _root32(row[0], "Gate 15 run genesis")
    prestart_root = _root32(row[1], "Gate 15 prestart receipt")
    endpoint_objects = _typed_tuple(row[2], 2, "Gate 15 endpoint rows")
    endpoints = tuple(_decode_endpoint_object(value) for value in endpoint_objects)
    if tuple(item.actor_id for item in endpoints) != _ENDPOINT_ACTORS:
        _fail("REJECT_Q1_GATE15", "endpoint order differs")
    if any(
        item.run_genesis_root != run_root
        or item.prestart_admission_receipt_root != prestart_root
        for item in endpoints
    ):
        _fail("REJECT_Q1_GATE15", "endpoint admission binding differs")
    return run_root, prestart_root, endpoints


def _gate16_subjects(
    subjects: object,
) -> tuple[bytes, bytes, bytes, tuple[Q1SignatureCompletionRowV1, ...], bytes]:
    row = _typed_tuple(subjects, 7, "Gate 16 subjects")
    python_root = _root32(row[0], "Gate 16 Python evidence")
    rust_root = _root32(row[1], "Gate 16 Rust evidence")
    file_root = _root32(row[2], "Gate 16 formal file closure")
    signature_rows = _decode_signature_rows(row[3])
    closure_root = _root32(row[4], "Gate 16 closure bundle")
    _exact_true(row[5], "Gate 16 formal byte equality")
    _exact_true(row[6], "Gate 16 state-preimage equality")
    return python_root, rust_root, file_root, signature_rows, closure_root


def _gate17_subjects(subjects: object) -> Q1DualReplayAgreementV1:
    row = _typed_tuple(subjects, 2, "Gate 17 subjects")
    agreement = _decode_dual_replay_object(row[0])
    if _root32(row[1], "Gate 17 agreement root") != agreement.agreement_root:
        _fail("REJECT_Q1_GATE17", "dual replay agreement root differs")
    return agreement


def _gate18_subjects(subjects: object) -> Q1TargetBlindAccessLedgerV1:
    row = _typed_tuple(subjects, 2, "Gate 18 subjects")
    ledger = _decode_access_ledger_object(row[0])
    if _root32(row[1], "Gate 18 access-ledger root") != ledger.ledger_root:
        _fail("REJECT_Q1_GATE18", "access-ledger root differs")
    return ledger


def _validate_gate_subjects_v1(gate_id: int, subjects: object) -> tuple[object, ...]:
    if type(gate_id) is not int or not 15 <= gate_id <= 20:
        _fail("REJECT_Q1_START_GATE", "gate ID is outside 15..20")
    row = subjects if type(subjects) is tuple else _typed_tuple(subjects, -1, "gate subjects")
    if gate_id == 15:
        _gate15_subjects(row)
    elif gate_id == 16:
        _gate16_subjects(row)
    elif gate_id == 17:
        _gate17_subjects(row)
    elif gate_id == 18:
        _gate18_subjects(row)
    elif gate_id == 19:
        checked = _typed_tuple(row, 4, "Gate 19 subjects")
        closure_root = _root32(checked[0], "Gate 19 closure bundle")
        core = _decode_completion_core_object(checked[1])
        if _root32(checked[2], "Gate 19 completion core") != core.core_root:
            _fail("REJECT_Q1_GATE19", "completion-core root differs")
        if not _wire_equal(checked[3], COMPLETION_SEAL_POLICY):
            _fail("REJECT_Q1_GATE19", "completion seal policy differs")
        if core.output_slots_1_through_7[4][2] != closure_root:
            _fail("REJECT_Q1_GATE19", "closure/core binding differs")
    else:
        checked = _typed_tuple(row, 3, "Gate 20 subjects")
        core_root = _root32(checked[0], "Gate 20 completion core")
        authority = _decode_complete_authority_object(checked[1])
        if not _wire_equal(checked[2], SLOT8_SELF_CONTENT_ROOT_POLICY):
            _fail("REJECT_Q1_GATE20", "slot-8 projection policy differs")
        if authority.completion_core_root != core_root:
            _fail("REJECT_Q1_GATE20", "authority/core binding differs")
    return row


def start_gate_evidence_root_v1(gate_id: int, subjects: tuple[object, ...]) -> bytes:
    checked = _validate_gate_subjects_v1(gate_id, subjects)
    registry_row = Q1_GATE_REGISTRY[gate_id - 1]
    return content_hash(
        START_GATE_EVIDENCE_ROOT_DOMAIN,
        (
            Q1_GATE_REGISTRY_ROOT,
            Q1_START_GATE_SUBJECT_REGISTRY_ROOT,
            gate_id,
            registry_row[1],
            checked,
        ),
    )


def start_gate_row_v1(gate_id: int, subjects: tuple[object, ...]) -> tuple[object, ...]:
    return (
        gate_id,
        Q1_GATE_REGISTRY[gate_id - 1][1],
        True,
        start_gate_evidence_root_v1(gate_id, subjects),
    )


def _validate_output_slots_1_through_7(
    value: object,
) -> tuple[tuple[int, bytes, bytes], ...]:
    rows = _typed_tuple(value, 7, "Q1 output slots 1 through 7")
    output: list[tuple[int, bytes, bytes]] = []
    for index, row in enumerate(rows, 1):
        checked = _typed_tuple(row, 3, f"Q1 output slot {index}")
        if (
            _uint(checked[0], "output-slot ordinal") != index
            or checked[1] != Q1_OUTPUT_SLOT_NAMES[index - 1]
        ):
            _fail("REJECT_Q1_COMPLETION_SLOTS", "output-slot identity differs")
        output.append((index, checked[1], _root32(checked[2], "output-slot root")))
    return tuple(output)


@dataclass(frozen=True, slots=True)
class Q1CompletionCoreV1:
    run_genesis_root: bytes
    prestart_admission_receipt_root: bytes
    gate_subject_rows_15_through_18: tuple[tuple[object, ...], ...]
    output_slots_1_through_7: tuple[tuple[int, bytes, bytes], ...]

    def __post_init__(self) -> None:
        _root32(self.run_genesis_root, "completion-core run genesis")
        _root32(self.prestart_admission_receipt_root, "completion-core prestart receipt")
        if type(self.gate_subject_rows_15_through_18) is not tuple or len(
            self.gate_subject_rows_15_through_18
        ) != 4:
            _fail("REJECT_Q1_COMPLETION_CORE", "four execution subject rows required")
        gate15 = _gate15_subjects(self.gate_subject_rows_15_through_18[0])
        gate16 = _gate16_subjects(self.gate_subject_rows_15_through_18[1])
        agreement = _gate17_subjects(self.gate_subject_rows_15_through_18[2])
        ledger = _gate18_subjects(self.gate_subject_rows_15_through_18[3])
        run_root, prestart_root, endpoints = gate15
        if run_root != self.run_genesis_root or prestart_root != self.prestart_admission_receipt_root:
            _fail("REJECT_Q1_COMPLETION_CORE", "core/prestart binding differs")
        python_root, rust_root, file_root, shared_rows, closure_root = gate16
        python, rust = endpoints
        if (
            python.evidence_root != python_root
            or rust.evidence_root != rust_root
            or python.signature_rows != shared_rows
            or rust.signature_rows != shared_rows
            or python.closure_bundle_root != closure_root
            or rust.closure_bundle_root != closure_root
        ):
            _fail("REJECT_Q1_COMPLETION_CORE", "Gate 15/16 equality binding differs")
        if (
            agreement.run_genesis_root != run_root
            or agreement.prestart_admission_receipt_root != prestart_root
            or agreement.python_endpoint_evidence_root != python_root
            or agreement.rust_endpoint_evidence_root != rust_root
            or agreement.formal_file_closure_root != file_root
            or agreement.signature_rows != shared_rows
            or agreement.closure_bundle_root != closure_root
        ):
            _fail("REJECT_Q1_COMPLETION_CORE", "Gate 16/17 replay binding differs")
        if (
            ledger.run_genesis_root != run_root
            or ledger.prestart_admission_receipt_root != prestart_root
            or ledger.dual_replay_agreement_root != agreement.agreement_root
            or ledger.actor_rows[0].actor_implementation_root
            != python.actor_implementation_root
            or ledger.actor_rows[1].actor_implementation_root
            != rust.actor_implementation_root
            or ledger.actor_rows[2].actor_implementation_root
            != agreement.host_actor_implementation_root
        ):
            _fail("REJECT_Q1_COMPLETION_CORE", "Gate 17/18 isolation binding differs")
        slots = _validate_output_slots_1_through_7(self.output_slots_1_through_7)
        expected_slots = (
            (1, Q1_OUTPUT_SLOT_NAMES[0], shared_rows[0].signature_manifest_root),
            (2, Q1_OUTPUT_SLOT_NAMES[1], shared_rows[0].saturation_state_root),
            (3, Q1_OUTPUT_SLOT_NAMES[2], shared_rows[1].signature_manifest_root),
            (4, Q1_OUTPUT_SLOT_NAMES[3], shared_rows[1].saturation_state_root),
            (5, Q1_OUTPUT_SLOT_NAMES[4], closure_root),
            (6, Q1_OUTPUT_SLOT_NAMES[5], agreement.agreement_root),
            (7, Q1_OUTPUT_SLOT_NAMES[6], ledger.ledger_root),
        )
        if slots != expected_slots:
            _fail("REJECT_Q1_COMPLETION_CORE", "output slots do not match replay roots")

    @property
    def gate_rows_15_through_18(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            start_gate_row_v1(gate_id, subjects)
            for gate_id, subjects in zip(
                range(15, 19), self.gate_subject_rows_15_through_18, strict=True
            )
        )

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q1_COMPLETION_CORE_TAG,
            COMPLETION_CORE_SCHEMA_ID,
            START_COMPLETION_TAG_REGISTRY_ROOT,
            Q1_GATE_REGISTRY_ROOT,
            Q1_START_GATE_SUBJECT_REGISTRY_ROOT,
            self.run_genesis_root,
            self.prestart_admission_receipt_root,
            self.gate_subject_rows_15_through_18,
            self.gate_rows_15_through_18,
            self.output_slots_1_through_7,
            b"NOT_RUN",
            None,
            False,
            None,
            False,
            False,
            False,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def core_root(self) -> bytes:
        return content_hash(COMPLETION_CORE_ROOT_DOMAIN, self.canonical_object())


def _decode_completion_core_object(value: object) -> Q1CompletionCoreV1:
    row = _typed_tuple(value, 18, "completion core")
    if (
        not _wire_equal(
            row[:6],
            (
                1,
                Q1_COMPLETION_CORE_TAG,
                COMPLETION_CORE_SCHEMA_ID,
                START_COMPLETION_TAG_REGISTRY_ROOT,
                Q1_GATE_REGISTRY_ROOT,
                Q1_START_GATE_SUBJECT_REGISTRY_ROOT,
            ),
        )
        or not _wire_equal(
            row[11:], (b"NOT_RUN", None, False, None, False, False, False)
        )
    ):
        _fail("REJECT_Q1_COMPLETION_CORE", "completion-core fixed wire differs")
    core = Q1CompletionCoreV1(
        run_genesis_root=row[6],
        prestart_admission_receipt_root=row[7],
        gate_subject_rows_15_through_18=row[8],
        output_slots_1_through_7=row[10],
    )
    if not _wire_equal(
        row[9], core.gate_rows_15_through_18
    ) or not _wire_equal(core.canonical_object(), row):
        _fail("REJECT_Q1_COMPLETION_CORE", "completion-core replay differs")
    return core


def decode_completion_core_v1(payload: bytes) -> Q1CompletionCoreV1:
    return _decode_completion_core_object(_strict_object(payload, "completion core"))


@dataclass(frozen=True, slots=True)
class Q1CompleteAuthorityV1:
    completion_core_root: bytes
    output_slots_1_through_7: tuple[tuple[int, bytes, bytes], ...]

    def __post_init__(self) -> None:
        _root32(self.completion_core_root, "complete-authority core")
        _validate_output_slots_1_through_7(self.output_slots_1_through_7)

    def canonical_object(self) -> tuple[object, ...]:
        projected_slots = self.output_slots_1_through_7 + (
            (8, Q1_OUTPUT_SLOT_NAMES[7], SLOT8_SELF_CONTENT_ROOT_POLICY),
        )
        return (
            1,
            COMPLETE_AUTHORITY_SCHEMA_ID,
            self.completion_core_root,
            b"COMPLETE",
            Q1_COMPLETE_GATE_COUNT,
            Q1_COMPLETE_GATE_MASK,
            Q1_COMPLETE_GATE_COUNT,
            8,
            projected_slots,
            b"NOT_RUN",
            None,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
        )


def _decode_complete_authority_object(value: object) -> Q1CompleteAuthorityV1:
    row = _typed_tuple(value, 19, "Q1 complete authority")
    if (
        not _wire_equal(row[:2], (1, COMPLETE_AUTHORITY_SCHEMA_ID))
        or not _wire_equal(
            row[3:8],
            (
                b"COMPLETE",
                Q1_COMPLETE_GATE_COUNT,
                Q1_COMPLETE_GATE_MASK,
                Q1_COMPLETE_GATE_COUNT,
                8,
            ),
        )
        or not _wire_equal(
            row[9:],
            (b"NOT_RUN", None, False, None, False, False, False, False, False, False),
        )
    ):
        _fail("REJECT_Q1_COMPLETE_AUTHORITY", "authority fixed wire differs")
    slot_rows = _typed_tuple(row[8], 8, "projected Q1 output slots")
    if not _wire_equal(
        slot_rows[7], (8, Q1_OUTPUT_SLOT_NAMES[7], SLOT8_SELF_CONTENT_ROOT_POLICY)
    ):
        _fail("REJECT_Q1_COMPLETE_AUTHORITY", "slot-8 self-root rule differs")
    authority = Q1CompleteAuthorityV1(row[2], _validate_output_slots_1_through_7(slot_rows[:7]))
    if not _wire_equal(authority.canonical_object(), row):
        _fail("REJECT_Q1_COMPLETE_AUTHORITY", "authority replay differs")
    return authority


@dataclass(frozen=True, slots=True)
class Q1CompletionReceiptV1:
    core: Q1CompletionCoreV1

    def __post_init__(self) -> None:
        if type(self.core) is not Q1CompletionCoreV1:
            _fail("REJECT_Q1_COMPLETION_RECEIPT", "completion core exact type differs")

    @property
    def authority(self) -> Q1CompleteAuthorityV1:
        return Q1CompleteAuthorityV1(self.core.core_root, self.core.output_slots_1_through_7)

    @property
    def gate19_subjects(self) -> tuple[object, ...]:
        return (
            self.core.output_slots_1_through_7[4][2],
            self.core.canonical_object(),
            self.core.core_root,
            COMPLETION_SEAL_POLICY,
        )

    @property
    def gate20_subjects(self) -> tuple[object, ...]:
        return (
            self.core.core_root,
            self.authority.canonical_object(),
            SLOT8_SELF_CONTENT_ROOT_POLICY,
        )

    @property
    def gate_rows_15_through_20(self) -> tuple[tuple[object, ...], ...]:
        return self.core.gate_rows_15_through_18 + (
            start_gate_row_v1(19, self.gate19_subjects),
            start_gate_row_v1(20, self.gate20_subjects),
        )

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q1_COMPLETION_RECEIPT_TAG,
            COMPLETION_RECEIPT_SCHEMA_ID,
            START_COMPLETION_TAG_REGISTRY_ROOT,
            Q1_GATE_REGISTRY_ROOT,
            Q1_START_GATE_SUBJECT_REGISTRY_ROOT,
            self.core.core_root,
            self.core.canonical_object(),
            Q1_START_GATE_COUNT,
            self.gate_rows_15_through_20,
            Q1_COMPLETE_GATE_COUNT,
            Q1_COMPLETE_GATE_MASK,
            True,
            self.authority.canonical_object(),
            SLOT8_SELF_CONTENT_ROOT_POLICY,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def receipt_root(self) -> bytes:
        return content_hash(COMPLETION_RECEIPT_ROOT_DOMAIN, self.canonical_object())

    @property
    def output_slots(self) -> tuple[tuple[int, bytes, bytes], ...]:
        return self.core.output_slots_1_through_7 + (
            (8, Q1_OUTPUT_SLOT_NAMES[7], self.receipt_root),
        )


def decode_completion_receipt_v1(payload: bytes) -> Q1CompletionReceiptV1:
    row = _strict_object(payload, "completion receipt")
    if (
        len(row) != 15
        or not _wire_equal(
            row[:6],
            (
                1,
                Q1_COMPLETION_RECEIPT_TAG,
                COMPLETION_RECEIPT_SCHEMA_ID,
                START_COMPLETION_TAG_REGISTRY_ROOT,
                Q1_GATE_REGISTRY_ROOT,
                Q1_START_GATE_SUBJECT_REGISTRY_ROOT,
            ),
        )
        or type(row[8]) is not int
        or row[8] != Q1_START_GATE_COUNT
        or not _wire_equal(
            row[10:13], (Q1_COMPLETE_GATE_COUNT, Q1_COMPLETE_GATE_MASK, True)
        )
        or not _wire_equal(row[14], SLOT8_SELF_CONTENT_ROOT_POLICY)
    ):
        _fail("REJECT_Q1_COMPLETION_RECEIPT", "completion-receipt fixed wire differs")
    core = _decode_completion_core_object(row[7])
    if row[6] != core.core_root:
        _fail("REJECT_Q1_COMPLETION_RECEIPT", "receipt/core root differs")
    receipt = Q1CompletionReceiptV1(core)
    if (
        not _wire_equal(row[9], receipt.gate_rows_15_through_20)
        or not _wire_equal(row[13], receipt.authority.canonical_object())
        or not _wire_equal(receipt.canonical_object(), row)
    ):
        _fail("REJECT_Q1_COMPLETION_RECEIPT", "completion-receipt replay differs")
    return receipt


def output_slots_1_through_7_v1(
    agreement: Q1DualReplayAgreementV1,
    ledger: Q1TargetBlindAccessLedgerV1,
) -> tuple[tuple[int, bytes, bytes], ...]:
    if type(agreement) is not Q1DualReplayAgreementV1 or type(
        ledger
    ) is not Q1TargetBlindAccessLedgerV1:
        _fail("REJECT_Q1_COMPLETION_SLOTS", "agreement/ledger exact types required")
    if ledger.dual_replay_agreement_root != agreement.agreement_root:
        _fail("REJECT_Q1_COMPLETION_SLOTS", "ledger/agreement binding differs")
    odd, sink = agreement.signature_rows
    return (
        (1, Q1_OUTPUT_SLOT_NAMES[0], odd.signature_manifest_root),
        (2, Q1_OUTPUT_SLOT_NAMES[1], odd.saturation_state_root),
        (3, Q1_OUTPUT_SLOT_NAMES[2], sink.signature_manifest_root),
        (4, Q1_OUTPUT_SLOT_NAMES[3], sink.saturation_state_root),
        (5, Q1_OUTPUT_SLOT_NAMES[4], agreement.closure_bundle_root),
        (6, Q1_OUTPUT_SLOT_NAMES[5], agreement.agreement_root),
        (7, Q1_OUTPUT_SLOT_NAMES[6], ledger.ledger_root),
    )


__all__ = [
    "COMPLETION_SEAL_POLICY",
    "ENDPOINT_RESULT_JSON_ROOT_DOMAIN",
    "FORMAL_INPUT_KEY_ABSENCE_SCHEMA_ID",
    "FORMAL_INPUT_KEY_ABSENCE_ROOT_DOMAIN",
    "FORMAL_FILE_CLOSURE_BYTES_DOMAIN",
    "HOST_RESULT_JSON_ROOT_DOMAIN",
    "ISOLATION_POSTCHECK_ROOT_DOMAIN",
    "ISOLATION_PRECHECK_ROOT_DOMAIN",
    "OUTPUT_FILE_CLOSURE_BYTES_DOMAIN",
    "RUNTIME_MOUNT_REGISTRY_SCHEMA_ID",
    "RUNTIME_MOUNT_REGISTRY_ROOT_DOMAIN",
    "Q1_COMPLETE_GATE_COUNT",
    "Q1_COMPLETE_GATE_MASK",
    "Q1_START_GATE_SUBJECT_REGISTRY",
    "Q1_START_GATE_SUBJECT_REGISTRY_ROOT",
    "Q1CompleteAuthorityV1",
    "Q1CompletionCoreV1",
    "Q1CompletionReceiptV1",
    "Q1DualReplayAgreementV1",
    "Q1EndpointCompletionEvidenceV1",
    "Q1FileEvidenceV1",
    "Q1IsolationEvidenceAdapterV1",
    "Q1IsolationPostcheckRowV1",
    "Q1MountManifestEvidenceRowV1",
    "Q1SignatureCompletionRowV1",
    "Q1SourceManifestEvidenceRowV1",
    "Q1StartCompletionWireError",
    "Q1TargetBlindAccessLedgerV1",
    "SLOT8_SELF_CONTENT_ROOT_POLICY",
    "START_COMPLETION_TAG_REGISTRY",
    "START_COMPLETION_TAG_REGISTRY_ROOT",
    "decode_completion_core_v1",
    "decode_completion_receipt_v1",
    "decode_dual_replay_agreement_v1",
    "decode_target_blind_access_ledger_v1",
    "endpoint_result_root_v1",
    "formal_input_key_registry_absence_root_v1",
    "formal_output_file_closure_root_v1",
    "host_replay_result_root_v1",
    "isolation_postcheck_root_v1",
    "isolation_precheck_root_v1",
    "mount_manifest_root_v1",
    "output_slots_1_through_7_v1",
    "output_file_closure_root_v1",
    "runtime_mount_registry_root_v1",
    "scratch_cleanup_root_v1",
    "signature_guard_ceiling_rows_v1",
    "source_module_closure_root_v1",
    "start_gate_evidence_root_v1",
    "start_gate_row_v1",
]
