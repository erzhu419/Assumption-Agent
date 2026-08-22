"""Verifier-only pure replay core for Phase-2B Step-5 CAS store/2.

This module mechanically reproduces the complete-session replay contract from
the frozen store/2 source whose SHA-256 is recorded below.  It consumes only
raw canonical artifacts: the out-of-band-pinned authority V2 graph, one
retention plan, and exact-order request, object, signed-receipt, checkpoint,
head, and response sequences.  It has no store service, filesystem, file
descriptor, socket, SCM_RIGHTS, signer, private key, clock, path, callback, or
decoded-object entry point.

Successful replay proves only the internal canonical-byte, content-address,
Ed25519-signature, chain, and IF_ABSENT/idempotent-complete mechanics of the
presented bytes.  Signed durable/readback and peer-UID fields are replayed as
signed claims; this pure process does not observe durability, a live service,
UID separation, key custody, or an external anti-rollback anchor.  It grants
neither authority nor readiness.

The complete helper graph, authority loaders, RFC8785 functions, concrete
Ed25519 backend, and result issuer are captured when this module is imported.
Ordinary module-global rebinding therefore cannot replace them.  Arbitrary
interpreter-memory or closure-cell mutation remains outside the claim boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import threading
import weakref
from typing import Final, Mapping, NoReturn

from .phase2b_formal_wire_covert_step5_authority_v2 import (
    ROLE_ORDER_V2 as _AUTHORITY_ROLE_ORDER_V2,
    ROLE_SIGNATURE_PURPOSES_V2 as _AUTHORITY_ROLE_SIGNATURE_PURPOSES_V2,
    SERVICE_ORDER_V2 as _AUTHORITY_SERVICE_ORDER_V2,
    load_step5_authority_profile_v2 as _authority_load_profile_v2,
    load_step5_signer_registry_v2 as _authority_load_registry_v2,
    load_step5_stage_b_trust_lock_v2 as _authority_load_lock_v2,
    phase2b_step5_role_key_id_v2 as _authority_role_key_id_v2,
    require_locked_step5_authority_graph_v2 as _authority_require_graph_v2,
)
from .rfc8785_v1 import (
    canonicalize_rfc8785_v1 as _canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1 as _decode_canonical_rfc8785_v1,
)

try:
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - pinned pure image must provide it.
    _InvalidSignature = None
    _Ed25519PublicKey = None


SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-store-replay-core/2"
)
RETENTION_PLAN_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-retention-plan/1"
)
GENERIC_STORE_REQUEST_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-generic-scm-request/1"
)
TRANSITION_BINDING_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-transition-binding/1"
)
STORE_RECEIPT_BODY_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-store-receipt-body/2"
)
SIGNED_STORE_RECEIPT_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-signed-cas-store-receipt/2"
)
STORE_CHECKPOINT_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-store-checkpoint/2"
)
STORE_HEAD_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-store-head/2"
)
STORE_RESPONSE_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-store-response/2"
)

CAS_STORE_ROLE: Final = "CAS_STORE"
CAS_STORE_CONTROL_DOMAIN: Final = "STEP5_CAS_STORE_CONTROL"
CAS_STORE_RECEIPT_PURPOSE: Final = "SIGN_CAS_IF_ABSENT_RECEIPT"
CAS_STORE_SIGNATURE_DOMAIN: Final = (
    "HEGEL_MACHINE_PHASE2B_FORMAL_WIRE_COVERT_STEP5_CAS_STORE_RECEIPT_V1"
)
CAS_STORE_CHECKPOINT_PURPOSE: Final = "CAS_STORE_APPEND_ONLY_CHECKPOINT_V2"
CAS_STORE_HEAD_PURPOSE: Final = "CAS_STORE_CONTENT_DERIVED_HEAD_V2"
CAS_STORE_RESPONSE_PURPOSE: Final = "CAS_STORE_DURABLE_RETENTION_RESPONSE_V2"
KEY_EPOCH: Final = 1

INSPECTED_STORE_V2_SOURCE_SHA256: Final = (
    "6c545144dd584244d9405c92120e09c8ecb1d115c90e964a75fb3239b0b670b1"
)
AUTHORITY_CLAIM_BOUNDARY: Final = (
    "CAS_STORE_V2_MECHANICS_ONLY_SAME_PROCESS_SAME_UID_LOCAL_FILESYSTEM_"
    "DECLARATIVE_SOURCE_PINS_AND_STATIC_AUTHORITY_V2_SERVICE_BINDINGS_NOT_LIVE_"
    "AND_PRESENTED_COMPLETE_REPLAY_ARE_NOT_AUTHORITY_"
    "NOT_IMAGE_OR_RUNTIME_CLOSURE_NOT_KEY_CUSTODY_NOT_ANTI_ROLLBACK_OR_ANTI_CLONE"
)
PURE_REPLAY_CLAIM_BOUNDARY: Final = (
    "PRESENTED_RAW_COMPLETE_SESSION_CANONICAL_CONTENT_SIGNATURE_CHAIN_AND_"
    "IF_ABSENT_IDEMPOTENCY_MECHANICS_ONLY_NOT_DURABILITY_NOT_LIVE_SERVICE_"
    "NOT_UID_SEPARATION_NOT_KEY_CUSTODY_NOT_EXTERNAL_ANTI_ROLLBACK_"
    "NOT_AUTHORITY_NOT_READINESS"
)
FAIL_CLOSED_ROLLBACK_DISCLOSURE: Final = (
    "GAPS_FORKS_STALE_CURRENT_HEAD_AND_PARTIAL_APPENDS_FAIL_CLOSED_BUT_A_"
    "CONSISTENT_WHOLE_TREE_ROLLBACK_OR_CLONE_REQUIRES_AN_EXTERNAL_ANCHOR_TO_DETECT"
)

MAXIMUM_PLAN_BYTES: Final = 4 * 1_024 * 1_024
MAXIMUM_REQUEST_BYTES: Final = 128 * 1_024
MAXIMUM_RECEIPT_BYTES: Final = 1 * 1_024 * 1_024
MAXIMUM_CHECKPOINT_BYTES: Final = 2 * 1_024 * 1_024
MAXIMUM_HEAD_BYTES: Final = 1 * 1_024 * 1_024
MAXIMUM_RESPONSE_BYTES: Final = 5 * 1_024 * 1_024
MAXIMUM_PLAN_OBJECTS: Final = 65_536
MAXIMUM_PLANNED_OBJECT_BYTES: Final = 1 << 40
GENESIS_CHECKPOINT_SHA256: Final = "sha256:" + ("0" * 64)

FAIL_INPUT: Final = "FAIL_STEP5_CAS_STORE_REPLAY_CORE_V2_INPUT"
FAIL_CANONICAL: Final = "FAIL_STEP5_CAS_STORE_REPLAY_CORE_V2_CANONICAL"
FAIL_SCHEMA: Final = "FAIL_STEP5_CAS_STORE_REPLAY_CORE_V2_SCHEMA"
FAIL_BINDING: Final = "FAIL_STEP5_CAS_STORE_REPLAY_CORE_V2_BINDING"
FAIL_AUTHORITY: Final = "FAIL_STEP5_CAS_STORE_REPLAY_CORE_V2_AUTHORITY"
FAIL_SIGNATURE: Final = "FAIL_STEP5_CAS_STORE_REPLAY_CORE_V2_SIGNATURE"
FAIL_CHAIN: Final = "FAIL_STEP5_CAS_STORE_REPLAY_CORE_V2_CHAIN"
FAIL_INCOMPLETE: Final = "FAIL_STEP5_CAS_STORE_REPLAY_CORE_V2_INCOMPLETE"


class CasStoreReplayCoreV2Error(ValueError):
    """Stable fail-closed error for pure store/2 complete-session replay."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _build_store_replay_core_v2() -> tuple[type[object], object, object]:
    """Construct the entire verifier graph around import-time captures."""

    error_type = CasStoreReplayCoreV2Error
    canonical_backend = _canonicalize_rfc8785_v1
    decode_backend = _decode_canonical_rfc8785_v1
    sha256_backend = hashlib.sha256
    authority_load_lock = _authority_load_lock_v2
    authority_load_profile = _authority_load_profile_v2
    authority_load_registry = _authority_load_registry_v2
    authority_require_graph = _authority_require_graph_v2
    authority_role_key_id = _authority_role_key_id_v2
    authority_role_order = tuple(_AUTHORITY_ROLE_ORDER_V2)
    authority_role_purposes = {
        role: tuple(_AUTHORITY_ROLE_SIGNATURE_PURPOSES_V2[role])
        for role in authority_role_order
    }
    authority_service_order = tuple(_AUTHORITY_SERVICE_ORDER_V2)
    ed25519_backend_loader = (
        None
        if _Ed25519PublicKey is None
        else _Ed25519PublicKey.from_public_bytes
    )
    invalid_signature_type = _InvalidSignature
    exact_type = type
    exact_len = len
    exact_tuple = tuple
    exact_list = list
    exact_dict = dict
    exact_bytes = bytes
    exact_str = str
    exact_int = int
    exact_bool = bool
    exact_float = float
    exact_frozenset = frozenset
    exact_all = all
    exact_any = any
    exact_zip = zip
    exact_enumerate = enumerate
    exact_range = range
    exact_abs = abs
    is_instance = isinstance
    exact_set = set
    exception_type = Exception
    attribute_error_type = AttributeError
    type_error_type = TypeError
    mapping_type = Mapping
    bytes_fromhex = bytes.fromhex
    weak_reference = weakref.ref
    identity_of = id
    get_attribute = getattr

    schema_plan = RETENTION_PLAN_SCHEMA_VERSION
    schema_request = GENERIC_STORE_REQUEST_SCHEMA_VERSION
    schema_transition = TRANSITION_BINDING_SCHEMA_VERSION
    schema_receipt_body = STORE_RECEIPT_BODY_SCHEMA_VERSION
    schema_signed_receipt = SIGNED_STORE_RECEIPT_SCHEMA_VERSION
    schema_checkpoint = STORE_CHECKPOINT_SCHEMA_VERSION
    schema_head = STORE_HEAD_SCHEMA_VERSION
    schema_response = STORE_RESPONSE_SCHEMA_VERSION
    cas_role = CAS_STORE_ROLE
    cas_control = CAS_STORE_CONTROL_DOMAIN
    cas_receipt_purpose = CAS_STORE_RECEIPT_PURPOSE
    cas_signature_domain = CAS_STORE_SIGNATURE_DOMAIN
    cas_checkpoint_purpose = CAS_STORE_CHECKPOINT_PURPOSE
    cas_head_purpose = CAS_STORE_HEAD_PURPOSE
    cas_response_purpose = CAS_STORE_RESPONSE_PURPOSE
    key_epoch = KEY_EPOCH
    store_claim_boundary = AUTHORITY_CLAIM_BOUNDARY
    pure_claim_boundary = PURE_REPLAY_CLAIM_BOUNDARY
    rollback_disclosure = FAIL_CLOSED_ROLLBACK_DISCLOSURE
    genesis_checkpoint_sha256 = GENESIS_CHECKPOINT_SHA256

    max_plan = MAXIMUM_PLAN_BYTES
    max_request = MAXIMUM_REQUEST_BYTES
    max_receipt = MAXIMUM_RECEIPT_BYTES
    max_checkpoint = MAXIMUM_CHECKPOINT_BYTES
    max_head = MAXIMUM_HEAD_BYTES
    max_response = MAXIMUM_RESPONSE_BYTES
    max_objects = MAXIMUM_PLAN_OBJECTS
    max_object_bytes = MAXIMUM_PLANNED_OBJECT_BYTES
    max_safe_integer = (1 << 53) - 1

    fail_input = FAIL_INPUT
    fail_canonical = FAIL_CANONICAL
    fail_schema = FAIL_SCHEMA
    fail_binding = FAIL_BINDING
    fail_authority = FAIL_AUTHORITY
    fail_signature = FAIL_SIGNATURE
    fail_chain = FAIL_CHAIN
    fail_incomplete = FAIL_INCOMPLETE

    plan_id_prefix = "phase2b_step5_retention_plan_v1_"
    request_id_prefix = "phase2b_step5_cas_request_v1_"
    head_state_id_prefix = "phase2b_step5_cas_head_state_v2_"
    plan_id_domain = b"HEGEL/PHASE2B/STEP5/CAS/RETENTION_PLAN/V1\x00"
    request_id_domain = b"HEGEL/PHASE2B/STEP5/CAS/GENERIC_REQUEST/V1\x00"
    head_state_id_domain = b"HEGEL/PHASE2B/STEP5/CAS/HEAD_STATE/V2\x00"
    signature_prefix = cas_signature_domain.encode("ascii") + b"\x00"

    identifier_re = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{7,511}")
    artifact_id_re = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{7,1023}")
    sha256_re = re.compile(r"sha256:[0-9a-f]{64}")
    hex32_re = re.compile(r"[0-9a-f]{64}")
    hex64_re = re.compile(r"[0-9a-f]{128}")
    profile_id_re = re.compile(
        r"phase2b_step5_authority_profile_v2_[0-9a-f]{64}"
    )
    registry_id_re = re.compile(
        r"phase2b_step5_signer_registry_v2_[0-9a-f]{64}"
    )
    role_key_id_re = re.compile(
        r"phase2b_step5_role_key_v1_[0-9a-f]{64}"
    )
    plan_id_re = re.compile(
        r"phase2b_step5_retention_plan_v1_[0-9a-f]{64}"
    )
    request_id_re = re.compile(
        r"phase2b_step5_cas_request_v1_[0-9a-f]{64}"
    )
    head_state_id_re = re.compile(
        r"phase2b_step5_cas_head_state_v2_[0-9a-f]{64}"
    )

    plan_object_keys = frozenset(
        {"ordinal", "artifact_kind", "maximum_bytes"}
    )
    plan_keys = frozenset(
        {
            "schema_version",
            "authority_profile_id",
            "signer_registry_id",
            "attempt_context_id",
            "retention_session_id",
            "chain_id",
            "plan_kind",
            "expected_object_count",
            "objects_in_exact_order",
            "complete_head_required",
            "plan_id",
        }
    )
    request_keys = frozenset(
        {
            "schema_version",
            "authority_profile_id",
            "signer_registry_id",
            "attempt_context_id",
            "retention_session_id",
            "chain_id",
            "plan_id",
            "request_id",
            "ordinal",
            "artifact_kind",
            "bound_artifact_id",
            "bound_artifact_sha256",
            "reported_size_bytes",
            "fd_count",
        }
    )
    transition_keys = frozenset(
        {
            "schema_version",
            "signer_role",
            "signer_control_domain",
            "signature_purpose",
            "authority_profile_id",
            "signer_registry_id",
            "key_epoch",
            "signer_key_id",
            "attempt_context_id",
            "retention_session_id",
            "chain_id",
            "plan_id",
            "plan_kind",
            "request_id",
            "ordinal",
            "artifact_kind",
            "bound_artifact_id",
            "bound_artifact_sha256",
            "object_id",
            "object_sha256",
            "object_size_bytes",
            "previous_checkpoint_sha256",
            "expected_object_count",
            "observed_object_count",
            "complete",
            "complete_head_required",
            "head_state_id",
        }
    )
    receipt_body_keys = frozenset(
        {
            "schema_version",
            "purpose",
            "signature_domain",
            "binding",
            "peer_pid",
            "peer_uid",
            "peer_gid",
            "object_disposition",
            "durable_before_return",
            "readback_verified",
            "claim_boundary",
        }
    )
    signed_receipt_keys = frozenset(
        {
            "schema_version",
            "purpose",
            "signature_domain",
            "binding",
            "signed_body",
            "signed_body_sha256",
            "signer_public_key_hex",
            "ed25519_signature_hex",
        }
    )
    checkpoint_keys = frozenset(
        {
            "schema_version",
            "purpose",
            "binding",
            "signed_receipt_sha256",
            "signed_receipt",
            "claim_boundary",
        }
    )
    head_keys = frozenset(
        {
            "schema_version",
            "purpose",
            "binding",
            "checkpoint_sha256",
            "signed_receipt_sha256",
            "claim_boundary",
            "rollback_disclosure",
        }
    )
    response_keys = frozenset(
        {
            "schema_version",
            "purpose",
            "binding",
            "checkpoint_sha256",
            "checkpoint_record",
            "head_sha256",
            "head_record",
            "durable_before_return",
            "readback_verified",
            "claim_boundary",
        }
    )

    def fail(code: str, detail: str) -> NoReturn:
        raise error_type(code, detail)

    def canonical(value: object) -> bytes:
        try:
            return canonical_backend(value)  # type: ignore[arg-type]
        except exception_type as exc:
            raise error_type(
                fail_canonical, "value is outside exact RFC8785"
            ) from exc

    def restore_integral_numbers(value: object) -> object:
        if (
            exact_type(value) is exact_float
            and value.is_integer()
            and exact_abs(value) <= max_safe_integer
        ):
            return exact_int(value)
        if exact_type(value) is exact_list:
            return [restore_integral_numbers(item) for item in value]
        if exact_type(value) is exact_dict:
            return {
                key: restore_integral_numbers(item)
                for key, item in value.items()
            }
        return value

    def decode(payload: bytes, maximum: int, label: str) -> object:
        if (
            exact_type(payload) is not exact_bytes
            or not 0 < exact_len(payload) <= maximum
        ):
            fail(fail_input, f"{label} bytes are outside the frozen bound")
        try:
            decoded = decode_backend(payload)
        except exception_type as exc:
            raise error_type(
                fail_canonical, f"{label} is not exact RFC8785 bytes"
            ) from exc
        restored = restore_integral_numbers(decoded)
        if canonical(restored) != payload:
            fail(
                fail_canonical,
                f"{label} differs after exact integer restoration",
            )
        return restored

    def exact_equal(left: object, right: object) -> bool:
        if exact_type(left) is not exact_type(right):
            return False
        if exact_type(left) is exact_dict:
            if exact_frozenset(left) != exact_frozenset(  # type: ignore[arg-type]
                right
            ):
                return False
            return exact_all(
                exact_equal(left[key], right[key])  # type: ignore[index]
                for key in left
            )
        if exact_type(left) is exact_list or exact_type(left) is exact_tuple:
            return exact_len(left) == exact_len(right) and exact_all(  # type: ignore[arg-type]
                exact_equal(a, b)
                for a, b in exact_zip(  # type: ignore[arg-type]
                    left, right, strict=True
                )
            )
        return left == right

    def exact_object(
        value: object, keys: frozenset[str], label: str
    ) -> dict[str, object]:
        if (
            exact_type(value) is not exact_dict
            or exact_frozenset(value) != keys
        ):
            fail(fail_schema, f"{label} exact field set differs")
        return value

    def wire_integer(
        value: object,
        label: str,
        *,
        minimum: int = 0,
        maximum: int = max_safe_integer,
    ) -> int:
        if exact_type(value) is not exact_int:
            fail(fail_schema, f"{label} is not one exact JSON integer")
        if not minimum <= value <= maximum:
            fail(fail_schema, f"{label} is outside its frozen bound")
        return value

    def literal(value: object, expected: str, label: str) -> str:
        if exact_type(value) is not exact_str or value != expected:
            fail(fail_schema, f"{label} differs")
        return value

    def pattern(
        value: object, expression: re.Pattern[str], label: str
    ) -> str:
        if (
            exact_type(value) is not exact_str
            or expression.fullmatch(value) is None
        ):
            fail(fail_schema, f"{label} syntax differs")
        return value

    def require_bool(value: object, label: str) -> bool:
        if exact_type(value) is not exact_bool:
            fail(fail_schema, f"{label} must be an exact boolean")
        return value

    def sha256_bytes(payload: bytes) -> str:
        if exact_type(payload) is not exact_bytes:
            fail(fail_input, "SHA-256 input must be exact bytes")
        return "sha256:" + sha256_backend(payload).hexdigest()

    def content_id(prefix: str, domain: bytes, subject: object) -> str:
        return prefix + sha256_backend(domain + canonical(subject)).hexdigest()

    def without(value: Mapping[str, object], key: str) -> dict[str, object]:
        return {name: item for name, item in value.items() if name != key}

    @dataclass(frozen=True, slots=True)
    class AuthorityBinding:
        authority_profile_id: str
        signer_registry_id: str
        signer_role: str
        control_domain: str
        signature_purpose: str
        key_epoch: int
        signer_key_id: str
        public_key_bytes: bytes

    def cas_store_role_key_id(public_key: bytes) -> str:
        if (
            exact_type(public_key) is not exact_bytes
            or exact_len(public_key) != 32
        ):
            fail(
                fail_authority,
                "CAS_STORE public key must be exact raw 32 bytes",
            )
        row = {
            "schema_version": "hegel-machine-phase2b-step5-role-key/1",
            "role": cas_role,
            "control_domain": cas_control,
            "authorized_signature_purposes": [cas_receipt_purpose],
            "algorithm": "Ed25519",
            "key_epoch": key_epoch,
            "public_key_hex": public_key.hex(),
        }
        try:
            return authority_role_key_id(row)
        except exception_type as exc:
            raise error_type(
                fail_authority, "CAS_STORE role-key derivation failed"
            ) from exc

    def load_authority_binding(
        *,
        stage_b_lock_payload: bytes,
        authority_profile_payload: bytes,
        signer_registry_payload: bytes,
        out_of_band_expected_lock_id_bytes: bytes,
        out_of_band_expected_lock_payload_sha256_digest: bytes,
    ) -> AuthorityBinding:
        raw = (
            stage_b_lock_payload,
            authority_profile_payload,
            signer_registry_payload,
            out_of_band_expected_lock_id_bytes,
            out_of_band_expected_lock_payload_sha256_digest,
        )
        if exact_any(
            exact_type(payload) is not exact_bytes for payload in raw
        ):
            fail(
                fail_input,
                "authority graph and out-of-band pins must be exact bytes",
            )
        try:
            lock = authority_load_lock(
                stage_b_lock_payload,
                out_of_band_expected_lock_id_bytes=(
                    out_of_band_expected_lock_id_bytes
                ),
                out_of_band_expected_payload_sha256_digest=(
                    out_of_band_expected_lock_payload_sha256_digest
                ),
            )
            profile = authority_load_profile(authority_profile_payload, lock)
            registry = authority_load_registry(
                signer_registry_payload, lock, profile
            )
            authority_require_graph(lock, profile, registry)
        except exception_type as exc:
            raise error_type(
                fail_authority,
                "raw Stage-B/profile/registry V2 graph rejected",
            ) from exc

        profile_id = pattern(
            profile.artifact_id, profile_id_re, "authority profile ID"
        )
        registry_id = pattern(
            registry.artifact_id, registry_id_re, "signer registry ID"
        )
        if (
            lock.body.get("authority_profile_id") != profile_id
            or lock.body.get("signer_registry_id") != registry_id
            or registry.body.get("authority_profile_id") != profile_id
        ):
            fail(fail_authority, "authority graph lineage is not closed")
        policy = profile.body.get("role_policy")
        if not is_instance(policy, mapping_type) or (
            exact_tuple(policy.get("roles_in_exact_order", ()))
            != authority_role_order
            or exact_type(policy.get("role_count")) is not exact_int
            or policy.get("role_count") != 8
            or policy.get("distinct_public_keys") is not True
            or policy.get("distinct_key_ids") is not True
            or policy.get("distinct_control_domains") is not True
            or exact_type(policy.get("key_epoch")) is not exact_int
            or policy.get("key_epoch") != key_epoch
            or exact_type(policy.get("purpose_policy_version"))
            is not exact_int
            or policy.get("purpose_policy_version") != 2
            or exact_tuple(policy.get("services_in_exact_order", ()))
            != authority_service_order
            or exact_type(policy.get("service_binding_count"))
            is not exact_int
            or policy.get("service_binding_count") != 5
            or policy.get("distinct_service_uids_gids") is not True
            or policy.get("static_service_identity_only") is not True
        ):
            fail(
                fail_authority,
                "authority profile exact role policy differs",
            )
        rows = registry.body.get("role_keys_in_exact_order")
        if (
            exact_type(rows) is not exact_tuple
            or exact_len(rows) != exact_len(authority_role_order)
        ):
            fail(
                fail_authority,
                "signer registry exact role cardinality differs",
            )
        for cascade_role in (
            "GOVERNANCE_ANCHOR",
            "SOURCE_RUNTIME_CUSTODIAN",
            "PURE_REPLAYER",
        ):
            row = rows[authority_role_order.index(cascade_role)]
            if (
                not is_instance(row, mapping_type)
                or exact_tuple(
                    row.get("authorized_signature_purposes", ())
                )
                != authority_role_purposes[cascade_role]
            ):
                fail(
                    fail_authority,
                    "authority V2 purpose cascade differs",
                )
        row = rows[authority_role_order.index(cas_role)]
        if not is_instance(row, mapping_type):
            fail(fail_authority, "CAS_STORE registry row type differs")
        purposes = row.get("authorized_signature_purposes")
        public_hex = row.get("public_key_hex")
        if (
            row.get("schema_version")
            != "hegel-machine-phase2b-step5-role-key/1"
            or row.get("role") != cas_role
            or row.get("control_domain") != cas_control
            or exact_type(purposes) is not exact_tuple
            or purposes != (cas_receipt_purpose,)
            or row.get("algorithm") != "Ed25519"
            or exact_type(row.get("key_epoch")) is not exact_int
            or row.get("key_epoch") != key_epoch
            or exact_type(public_hex) is not exact_str
            or hex32_re.fullmatch(public_hex) is None
        ):
            fail(
                fail_authority,
                "CAS_STORE registry policy/public key differs",
            )
        public_key = bytes_fromhex(public_hex)
        signer_key_id = pattern(
            row.get("key_id"), role_key_id_re, "CAS_STORE role-key ID"
        )
        if signer_key_id != cas_store_role_key_id(public_key):
            fail(
                fail_authority,
                "CAS_STORE registry key ID/content differs",
            )
        return AuthorityBinding(
            authority_profile_id=profile_id,
            signer_registry_id=registry_id,
            signer_role=cas_role,
            control_domain=cas_control,
            signature_purpose=cas_receipt_purpose,
            key_epoch=key_epoch,
            signer_key_id=signer_key_id,
            public_key_bytes=public_key,
        )

    def parse_plan(payload: bytes) -> dict[str, object]:
        body = exact_object(
            decode(payload, max_plan, "retention plan"),
            plan_keys,
            "retention plan",
        )
        literal(body["schema_version"], schema_plan, "retention plan schema")
        pattern(
            body["authority_profile_id"],
            profile_id_re,
            "retention plan authority profile ID",
        )
        pattern(
            body["signer_registry_id"],
            registry_id_re,
            "retention plan signer registry ID",
        )
        for field, label in (
            ("attempt_context_id", "retention plan attempt context ID"),
            ("retention_session_id", "retention plan session ID"),
            ("chain_id", "retention plan chain ID"),
            ("plan_kind", "retention plan kind"),
        ):
            pattern(body[field], identifier_re, label)
        expected = wire_integer(
            body["expected_object_count"],
            "retention plan expected object count",
            minimum=1,
            maximum=max_objects,
        )
        objects = body["objects_in_exact_order"]
        if (
            exact_type(objects) is not exact_list
            or exact_len(objects) != expected
        ):
            fail(
                fail_binding,
                "retention plan object cardinality differs",
            )
        for index, raw in exact_enumerate(objects):
            item = exact_object(raw, plan_object_keys, "planned object")
            if wire_integer(
                item["ordinal"],
                "planned object ordinal",
                maximum=max_objects - 1,
            ) != index:
                fail(
                    fail_binding,
                    "retention plan ordinals are not exact contiguous order",
                )
            pattern(
                item["artifact_kind"],
                identifier_re,
                "planned artifact kind",
            )
            wire_integer(
                item["maximum_bytes"],
                "planned object maximum bytes",
                minimum=1,
                maximum=max_object_bytes,
            )
        if require_bool(
            body["complete_head_required"],
            "retention plan complete-head requirement",
        ) is not True:
            fail(
                fail_binding,
                "retention plan/1 requires an exact complete head",
            )
        pattern(body["plan_id"], plan_id_re, "retention plan ID")
        expected_id = content_id(
            plan_id_prefix,
            plan_id_domain,
            without(body, "plan_id"),
        )
        if body["plan_id"] != expected_id:
            fail(
                fail_binding,
                "retention plan ID does not bind exact content",
            )
        return body

    def require_request_plan(
        request: Mapping[str, object], plan: Mapping[str, object]
    ) -> None:
        ordinal = request["ordinal"]
        if exact_type(ordinal) is not exact_int:
            fail(fail_schema, "request ordinal is not exact integer")
        objects = plan["objects_in_exact_order"]
        if (
            exact_type(objects) is not exact_list
            or not 0 <= ordinal < exact_len(objects)
        ):
            fail(fail_binding, "request ordinal is outside retention plan")
        entry = objects[ordinal]
        if exact_type(entry) is not exact_dict:
            fail(fail_schema, "planned object type differs")
        if (
            request["authority_profile_id"] != plan["authority_profile_id"]
            or request["signer_registry_id"] != plan["signer_registry_id"]
            or request["attempt_context_id"] != plan["attempt_context_id"]
            or request["retention_session_id"]
            != plan["retention_session_id"]
            or request["chain_id"] != plan["chain_id"]
            or request["plan_id"] != plan["plan_id"]
            or request["artifact_kind"] != entry["artifact_kind"]
            or request["reported_size_bytes"] > entry["maximum_bytes"]  # type: ignore[operator]
        ):
            fail(
                fail_binding,
                "generic request differs from exact retention plan",
            )

    def parse_request(
        payload: bytes, plan: Mapping[str, object]
    ) -> dict[str, object]:
        body = exact_object(
            decode(payload, max_request, "generic request"),
            request_keys,
            "generic store request",
        )
        literal(body["schema_version"], schema_request, "request schema")
        pattern(
            body["authority_profile_id"],
            profile_id_re,
            "request profile ID",
        )
        pattern(
            body["signer_registry_id"],
            registry_id_re,
            "request registry ID",
        )
        for field, expression, label in (
            ("attempt_context_id", identifier_re, "request attempt context ID"),
            ("retention_session_id", identifier_re, "request session ID"),
            ("chain_id", identifier_re, "request chain ID"),
            ("plan_id", plan_id_re, "request plan ID"),
            ("request_id", request_id_re, "request ID"),
            ("artifact_kind", identifier_re, "request artifact kind"),
            ("bound_artifact_id", artifact_id_re, "bound artifact ID"),
            ("bound_artifact_sha256", sha256_re, "bound artifact SHA-256"),
        ):
            pattern(body[field], expression, label)
        wire_integer(
            body["ordinal"],
            "request ordinal",
            maximum=max_objects - 1,
        )
        wire_integer(
            body["reported_size_bytes"],
            "reported object size",
            minimum=1,
            maximum=max_object_bytes,
        )
        if wire_integer(
            body["fd_count"], "request FD count", minimum=1, maximum=1
        ) != 1:
            fail(fail_binding, "generic request must declare exactly one FD")
        expected_id = content_id(
            request_id_prefix,
            request_id_domain,
            without(body, "request_id"),
        )
        if body["request_id"] != expected_id:
            fail(
                fail_binding,
                "request ID does not bind exact request fields",
            )
        require_request_plan(body, plan)
        return body

    def transition_to_request(
        binding: Mapping[str, object]
    ) -> dict[str, object]:
        return {
            "schema_version": schema_request,
            "authority_profile_id": binding["authority_profile_id"],
            "signer_registry_id": binding["signer_registry_id"],
            "attempt_context_id": binding["attempt_context_id"],
            "retention_session_id": binding["retention_session_id"],
            "chain_id": binding["chain_id"],
            "plan_id": binding["plan_id"],
            "request_id": binding["request_id"],
            "ordinal": binding["ordinal"],
            "artifact_kind": binding["artifact_kind"],
            "bound_artifact_id": binding["bound_artifact_id"],
            "bound_artifact_sha256": binding[
                "bound_artifact_sha256"
            ],
            "reported_size_bytes": binding["object_size_bytes"],
            "fd_count": 1,
        }

    def require_transition_authority(
        binding: Mapping[str, object], authority_binding: AuthorityBinding
    ) -> None:
        if (
            binding["signer_role"] != authority_binding.signer_role
            or binding["signer_control_domain"]
            != authority_binding.control_domain
            or binding["signature_purpose"]
            != authority_binding.signature_purpose
            or binding["authority_profile_id"]
            != authority_binding.authority_profile_id
            or binding["signer_registry_id"]
            != authority_binding.signer_registry_id
            or binding["key_epoch"] != authority_binding.key_epoch
            or binding["signer_key_id"]
            != authority_binding.signer_key_id
        ):
            fail(
                fail_authority,
                "transition differs from pinned CAS_STORE authority",
            )

    def parse_transition(
        value: object,
        plan: Mapping[str, object],
        authority_binding: AuthorityBinding,
    ) -> dict[str, object]:
        body = exact_object(
            value, transition_keys, "transition binding"
        )
        literal(body["schema_version"], schema_transition, "transition schema")
        literal(body["signer_role"], cas_role, "transition signer role")
        literal(
            body["signer_control_domain"],
            cas_control,
            "transition signer control domain",
        )
        literal(
            body["signature_purpose"],
            cas_receipt_purpose,
            "transition signature purpose",
        )
        for field, expression, label in (
            ("authority_profile_id", profile_id_re, "transition profile ID"),
            ("signer_registry_id", registry_id_re, "transition registry ID"),
            ("signer_key_id", role_key_id_re, "transition signer key ID"),
            ("attempt_context_id", identifier_re, "transition attempt ID"),
            ("retention_session_id", identifier_re, "transition session ID"),
            ("chain_id", identifier_re, "transition chain ID"),
            ("plan_id", plan_id_re, "transition plan ID"),
            ("plan_kind", identifier_re, "transition plan kind"),
            ("request_id", request_id_re, "transition request ID"),
            ("artifact_kind", identifier_re, "transition artifact kind"),
            ("bound_artifact_id", artifact_id_re, "transition artifact ID"),
            (
                "bound_artifact_sha256",
                sha256_re,
                "transition bound SHA-256",
            ),
            ("object_id", sha256_re, "transition object ID"),
            ("object_sha256", sha256_re, "transition object SHA-256"),
            (
                "previous_checkpoint_sha256",
                sha256_re,
                "transition predecessor checkpoint",
            ),
            ("head_state_id", head_state_id_re, "transition head state ID"),
        ):
            pattern(body[field], expression, label)
        if wire_integer(
            body["key_epoch"],
            "transition key epoch",
            minimum=1,
            maximum=1,
        ) != key_epoch:
            fail(fail_binding, "transition key epoch differs")
        ordinal = wire_integer(
            body["ordinal"],
            "transition ordinal",
            maximum=max_objects - 1,
        )
        wire_integer(
            body["object_size_bytes"],
            "transition object size",
            minimum=1,
            maximum=max_object_bytes,
        )
        expected = wire_integer(
            body["expected_object_count"],
            "transition expected object count",
            minimum=1,
            maximum=max_objects,
        )
        observed = wire_integer(
            body["observed_object_count"],
            "transition observed object count",
            minimum=1,
            maximum=max_objects,
        )
        complete = require_bool(
            body["complete"], "transition complete flag"
        )
        if require_bool(
            body["complete_head_required"],
            "transition complete-head requirement",
        ) is not True:
            fail(
                fail_binding,
                "transition does not require an exact complete head",
            )
        if (
            body["object_id"] != body["object_sha256"]
            or body["object_sha256"] != body["bound_artifact_sha256"]
        ):
            fail(
                fail_binding,
                "actual object ID/SHA differs from bound artifact SHA",
            )
        if (
            observed != ordinal + 1
            or observed > expected
            or complete is not (observed == expected)
        ):
            fail(
                fail_binding,
                "transition ordinal/count/complete linkage differs",
            )
        if ordinal == 0:
            if (
                body["previous_checkpoint_sha256"]
                != genesis_checkpoint_sha256
            ):
                fail(
                    fail_chain,
                    "first transition predecessor is not genesis",
                )
        elif (
            body["previous_checkpoint_sha256"]
            == genesis_checkpoint_sha256
        ):
            fail(
                fail_chain,
                "non-genesis transition uses genesis predecessor",
            )
        expected_head_id = content_id(
            head_state_id_prefix,
            head_state_id_domain,
            without(body, "head_state_id"),
        )
        if body["head_state_id"] != expected_head_id:
            fail(
                fail_binding,
                "head state ID does not bind exact transition content",
            )
        require_transition_authority(body, authority_binding)
        request = transition_to_request(body)
        require_request_plan(request, plan)
        if (
            body["plan_kind"] != plan["plan_kind"]
            or body["expected_object_count"]
            != plan["expected_object_count"]
            or body["complete_head_required"]
            != plan["complete_head_required"]
        ):
            fail(fail_binding, "transition plan summary differs")
        return body

    def parse_receipt_body(
        value: object,
        plan: Mapping[str, object],
        authority_binding: AuthorityBinding,
    ) -> dict[str, object]:
        body = exact_object(value, receipt_body_keys, "receipt body")
        literal(
            body["schema_version"],
            schema_receipt_body,
            "receipt body schema",
        )
        literal(body["purpose"], cas_receipt_purpose, "receipt body purpose")
        literal(
            body["signature_domain"],
            cas_signature_domain,
            "receipt signature domain",
        )
        binding = parse_transition(
            body["binding"], plan, authority_binding
        )
        wire_integer(body["peer_pid"], "receipt peer PID", minimum=1)
        wire_integer(body["peer_uid"], "receipt peer UID")
        wire_integer(body["peer_gid"], "receipt peer GID")
        if (
            exact_type(body["object_disposition"]) is not exact_str
            or body["object_disposition"]
            not in {"CREATED_IF_ABSENT", "PRESENT_VERIFIED"}
        ):
            fail(
                fail_binding,
                "receipt object disposition differs",
            )
        if (
            require_bool(
                body["durable_before_return"], "receipt durability claim"
            )
            is not True
            or require_bool(
                body["readback_verified"], "receipt readback claim"
            )
            is not True
        ):
            fail(
                fail_binding,
                "receipt lacks signed durable/readback claim",
            )
        literal(
            body["claim_boundary"],
            store_claim_boundary,
            "receipt claim boundary",
        )
        if not exact_equal(body["binding"], binding):
            fail(fail_binding, "receipt binding normalization differs")
        return body

    def parse_signed_receipt(
        value: object,
        plan: Mapping[str, object],
        authority_binding: AuthorityBinding,
    ) -> dict[str, object]:
        body = exact_object(
            value, signed_receipt_keys, "signed receipt"
        )
        literal(
            body["schema_version"],
            schema_signed_receipt,
            "signed receipt schema",
        )
        literal(body["purpose"], cas_receipt_purpose, "signed receipt purpose")
        literal(
            body["signature_domain"],
            cas_signature_domain,
            "signed receipt domain",
        )
        binding = parse_transition(
            body["binding"], plan, authority_binding
        )
        signed_body = parse_receipt_body(
            body["signed_body"], plan, authority_binding
        )
        if not exact_equal(binding, signed_body["binding"]):
            fail(
                fail_binding,
                "signed receipt/body transition bindings differ",
            )
        pattern(
            body["signed_body_sha256"],
            sha256_re,
            "signed receipt body SHA-256",
        )
        if body["signed_body_sha256"] != sha256_bytes(
            canonical(signed_body)
        ):
            fail(
                fail_binding,
                "signed receipt body hash differs",
            )
        public_hex = pattern(
            body["signer_public_key_hex"],
            hex32_re,
            "receipt signer public key",
        )
        signature_hex = pattern(
            body["ed25519_signature_hex"],
            hex64_re,
            "receipt Ed25519 signature",
        )
        if public_hex != authority_binding.public_key_bytes.hex():
            fail(
                fail_authority,
                "receipt public key differs from pinned registry",
            )
        if (
            ed25519_backend_loader is None
            or invalid_signature_type is None
        ):
            fail(
                fail_signature,
                "cryptography Ed25519 verifier is unavailable",
            )
        try:
            ed25519_backend_loader(
                authority_binding.public_key_bytes
            ).verify(
                bytes_fromhex(signature_hex),
                signature_prefix + canonical(signed_body),
            )
        except invalid_signature_type as exc:
            raise error_type(
                fail_signature,
                "CAS_STORE Ed25519 receipt rejected",
            ) from exc
        return body

    def parse_checkpoint(
        payload_or_value: object,
        *,
        raw: bool,
        plan: Mapping[str, object],
        authority_binding: AuthorityBinding,
    ) -> dict[str, object]:
        value = (
            decode(payload_or_value, max_checkpoint, "checkpoint")
            if raw
            else payload_or_value
        )
        body = exact_object(value, checkpoint_keys, "checkpoint")
        literal(
            body["schema_version"], schema_checkpoint, "checkpoint schema"
        )
        literal(
            body["purpose"],
            cas_checkpoint_purpose,
            "checkpoint purpose",
        )
        binding = parse_transition(
            body["binding"], plan, authority_binding
        )
        receipt = parse_signed_receipt(
            body["signed_receipt"], plan, authority_binding
        )
        pattern(
            body["signed_receipt_sha256"],
            sha256_re,
            "checkpoint receipt SHA-256",
        )
        if (
            not exact_equal(binding, receipt["binding"])
            or body["signed_receipt_sha256"]
            != sha256_bytes(canonical(receipt))
        ):
            fail(
                fail_binding,
                "checkpoint/signed-receipt linkage differs",
            )
        literal(
            body["claim_boundary"],
            store_claim_boundary,
            "checkpoint claim boundary",
        )
        return body

    def parse_head(
        payload_or_value: object,
        *,
        raw: bool,
        plan: Mapping[str, object],
        authority_binding: AuthorityBinding,
    ) -> dict[str, object]:
        value = (
            decode(payload_or_value, max_head, "head")
            if raw
            else payload_or_value
        )
        body = exact_object(value, head_keys, "head")
        literal(body["schema_version"], schema_head, "head schema")
        literal(body["purpose"], cas_head_purpose, "head purpose")
        parse_transition(body["binding"], plan, authority_binding)
        pattern(
            body["checkpoint_sha256"],
            sha256_re,
            "head checkpoint SHA-256",
        )
        pattern(
            body["signed_receipt_sha256"],
            sha256_re,
            "head receipt SHA-256",
        )
        literal(
            body["claim_boundary"],
            store_claim_boundary,
            "head claim boundary",
        )
        literal(
            body["rollback_disclosure"],
            rollback_disclosure,
            "head rollback disclosure",
        )
        return body

    def parse_response(
        payload_or_value: object,
        *,
        raw: bool,
        plan: Mapping[str, object],
        authority_binding: AuthorityBinding,
    ) -> dict[str, object]:
        value = (
            decode(payload_or_value, max_response, "response")
            if raw
            else payload_or_value
        )
        body = exact_object(value, response_keys, "response")
        literal(body["schema_version"], schema_response, "response schema")
        literal(
            body["purpose"],
            cas_response_purpose,
            "response purpose",
        )
        binding = parse_transition(
            body["binding"], plan, authority_binding
        )
        checkpoint = parse_checkpoint(
            body["checkpoint_record"],
            raw=False,
            plan=plan,
            authority_binding=authority_binding,
        )
        head = parse_head(
            body["head_record"],
            raw=False,
            plan=plan,
            authority_binding=authority_binding,
        )
        pattern(
            body["checkpoint_sha256"],
            sha256_re,
            "response checkpoint SHA-256",
        )
        pattern(
            body["head_sha256"],
            sha256_re,
            "response head SHA-256",
        )
        if (
            not exact_equal(binding, checkpoint["binding"])
            or not exact_equal(binding, head["binding"])
            or body["checkpoint_sha256"]
            != sha256_bytes(canonical(checkpoint))
            or body["head_sha256"] != sha256_bytes(canonical(head))
            or head["checkpoint_sha256"] != body["checkpoint_sha256"]
            or head["signed_receipt_sha256"]
            != checkpoint["signed_receipt_sha256"]
        ):
            fail(
                fail_binding,
                "response/checkpoint/head linkage differs",
            )
        if (
            require_bool(
                body["durable_before_return"], "response durability claim"
            )
            is not True
            or require_bool(
                body["readback_verified"], "response readback claim"
            )
            is not True
        ):
            fail(
                fail_binding,
                "response lacks signed durable/readback claim",
            )
        literal(
            body["claim_boundary"],
            store_claim_boundary,
            "response claim boundary",
        )
        return body

    def verify_triplet(
        *,
        checkpoint: Mapping[str, object],
        head: Mapping[str, object],
        response: Mapping[str, object],
    ) -> Mapping[str, object]:
        binding = checkpoint["binding"]
        if exact_type(binding) is not exact_dict:
            fail(fail_schema, "checkpoint binding type differs")
        if (
            not exact_equal(head["binding"], binding)
            or not exact_equal(response["binding"], binding)
            or not exact_equal(
                response["checkpoint_record"], checkpoint
            )
            or not exact_equal(response["head_record"], head)
            or head["checkpoint_sha256"]
            != sha256_bytes(canonical(checkpoint))
            or head["signed_receipt_sha256"]
            != checkpoint["signed_receipt_sha256"]
            or response["checkpoint_sha256"]
            != sha256_bytes(canonical(checkpoint))
            or response["head_sha256"] != sha256_bytes(canonical(head))
        ):
            fail(
                fail_binding,
                "record triplet exact linkage differs",
            )
        return binding

    issuance_token = object()
    issued_lock = threading.Lock()
    issued_results: dict[
        int, tuple[weakref.ReferenceType[object], tuple[object, ...]]
    ] = {}
    immutable_field_names = (
        "verification_status",
        "authority_profile_id",
        "signer_registry_id",
        "signer_role",
        "signer_key_id",
        "key_epoch",
        "attempt_context_id",
        "retention_session_id",
        "chain_id",
        "plan_id",
        "plan_kind",
        "expected_object_count",
        "complete_head_state_id",
        "complete_head_sha256",
        "complete_checkpoint_sha256",
        "complete_response_sha256",
        "object_bindings_in_exact_order",
        "object_dispositions_in_exact_order",
        "canonical_bytes_verified",
        "content_addresses_verified",
        "ed25519_signatures_verified",
        "append_chain_verified",
        "if_absent_idempotent_complete_semantics_verified",
        "durability_verified",
        "live_cas_store_verified",
        "service_image_verified",
        "runtime_closure_verified",
        "live_uid_gid_verified",
        "live_control_domain_verified",
        "private_key_custody_verified",
        "external_anti_rollback_verified",
        "authority_evidence",
        "readiness_evidence",
        "execution_authorized",
        "claim_boundary",
        "rollback_disclosure",
    )
    object_new = object.__new__
    object_setattr = object.__setattr__

    class _WeakReferenceableIssuedResult:
        __slots__ = ("__weakref__",)

    @dataclass(frozen=True, slots=True, init=False)
    class VerifiedCasStoreCompleteSessionReplayCoreV2(
        _WeakReferenceableIssuedResult
    ):
        """Privately issued result of raw pure complete-session replay."""

        verification_status: str
        authority_profile_id: str
        signer_registry_id: str
        signer_role: str
        signer_key_id: str
        key_epoch: int
        attempt_context_id: str
        retention_session_id: str
        chain_id: str
        plan_id: str
        plan_kind: str
        expected_object_count: int
        complete_head_state_id: str
        complete_head_sha256: str
        complete_checkpoint_sha256: str
        complete_response_sha256: str
        object_bindings_in_exact_order: tuple[tuple[object, ...], ...]
        object_dispositions_in_exact_order: tuple[str, ...]
        canonical_bytes_verified: bool
        content_addresses_verified: bool
        ed25519_signatures_verified: bool
        append_chain_verified: bool
        if_absent_idempotent_complete_semantics_verified: bool
        durability_verified: bool
        live_cas_store_verified: bool
        service_image_verified: bool
        runtime_closure_verified: bool
        live_uid_gid_verified: bool
        live_control_domain_verified: bool
        private_key_custody_verified: bool
        external_anti_rollback_verified: bool
        authority_evidence: bool
        readiness_evidence: bool
        execution_authorized: bool
        claim_boundary: str
        rollback_disclosure: str

        def __new__(
            cls, token: object = None
        ) -> "VerifiedCasStoreCompleteSessionReplayCoreV2":
            if token is not issuance_token:
                raise type_error_type(
                    "verified CAS replay-core results are privately issued"
                )
            return object_new(cls)

        def __copy__(
            self,
        ) -> "VerifiedCasStoreCompleteSessionReplayCoreV2":
            raise type_error_type(
                "verified CAS replay-core results cannot be copied"
            )

        def __deepcopy__(
            self, memo: object
        ) -> "VerifiedCasStoreCompleteSessionReplayCoreV2":
            raise type_error_type(
                "verified CAS replay-core results cannot be deep-copied"
            )

    def snapshot_result(
        result: VerifiedCasStoreCompleteSessionReplayCoreV2,
    ) -> tuple[object, ...]:
        try:
            return exact_tuple(
                get_attribute(result, name)
                for name in immutable_field_names
            )
        except (attribute_error_type, type_error_type) as exc:
            raise error_type(
                fail_binding,
                "verified replay-core result fields are missing",
            ) from exc

    def register_result(
        result: VerifiedCasStoreCompleteSessionReplayCoreV2,
    ) -> None:
        identity = identity_of(result)

        def forget(reference: weakref.ReferenceType[object]) -> None:
            with issued_lock:
                current = issued_results.get(identity)
                if current is not None and current[0] is reference:
                    del issued_results[identity]

        reference = weak_reference(result, forget)
        snapshot = snapshot_result(result)
        with issued_lock:
            if identity in issued_results:
                fail(
                    fail_binding,
                    "verified replay-core result identity unexpectedly reused",
                )
            issued_results[identity] = (reference, snapshot)

    def require_verified_cas_store_complete_session_replay_core_v2(
        candidate: object,
    ) -> VerifiedCasStoreCompleteSessionReplayCoreV2:
        """Require exact privately issued identity and unmodified fields."""

        if (
            exact_type(candidate)
            is not VerifiedCasStoreCompleteSessionReplayCoreV2
        ):
            fail(
                fail_input,
                "replay-core result has the wrong exact type",
            )
        identity = identity_of(candidate)
        with issued_lock:
            issued = issued_results.get(identity)
            if issued is None or issued[0]() is not candidate:
                fail(
                    fail_input,
                    "replay-core result was not privately issued",
                )
            frozen_snapshot = issued[1]
        if not exact_equal(snapshot_result(candidate), frozen_snapshot):
            fail(
                fail_binding,
                "issued replay-core result was mutated",
            )
        return candidate

    def issue_result(
        *,
        plan: Mapping[str, object],
        authority_binding: AuthorityBinding,
        records: tuple[
            tuple[
                Mapping[str, object],
                Mapping[str, object],
                Mapping[str, object],
            ],
            ...,
        ],
    ) -> VerifiedCasStoreCompleteSessionReplayCoreV2:
        checkpoint, head, response = records[-1]
        head_binding = head["binding"]
        if exact_type(head_binding) is not exact_dict:
            fail(fail_schema, "complete head binding type differs")
        result = VerifiedCasStoreCompleteSessionReplayCoreV2(issuance_token)
        values: dict[str, object] = {
            "verification_status": (
                "PRESENTED_RAW_COMPLETE_CAS_SESSION_REPLAY_CORE_VERIFIED_"
                "MECHANICS_ONLY_NOT_AUTHORITY"
            ),
            "authority_profile_id": plan["authority_profile_id"],
            "signer_registry_id": plan["signer_registry_id"],
            "signer_role": authority_binding.signer_role,
            "signer_key_id": authority_binding.signer_key_id,
            "key_epoch": authority_binding.key_epoch,
            "attempt_context_id": plan["attempt_context_id"],
            "retention_session_id": plan["retention_session_id"],
            "chain_id": plan["chain_id"],
            "plan_id": plan["plan_id"],
            "plan_kind": plan["plan_kind"],
            "expected_object_count": plan["expected_object_count"],
            "complete_head_state_id": head_binding["head_state_id"],
            "complete_head_sha256": sha256_bytes(canonical(head)),
            "complete_checkpoint_sha256": sha256_bytes(
                canonical(checkpoint)
            ),
            "complete_response_sha256": sha256_bytes(
                canonical(response)
            ),
            "object_bindings_in_exact_order": exact_tuple(
                (
                    record_checkpoint["binding"]["ordinal"],  # type: ignore[index]
                    record_checkpoint["binding"]["artifact_kind"],  # type: ignore[index]
                    record_checkpoint["binding"]["bound_artifact_id"],  # type: ignore[index]
                    record_checkpoint["binding"]["bound_artifact_sha256"],  # type: ignore[index]
                    record_checkpoint["binding"]["object_id"],  # type: ignore[index]
                    record_checkpoint["binding"]["object_size_bytes"],  # type: ignore[index]
                    record_checkpoint["signed_receipt_sha256"],
                    sha256_bytes(canonical(record_checkpoint)),
                    sha256_bytes(canonical(record_head)),
                    sha256_bytes(canonical(record_response)),
                )
                for record_checkpoint, record_head, record_response in records
            ),
            "object_dispositions_in_exact_order": exact_tuple(
                record_checkpoint["signed_receipt"]["signed_body"][  # type: ignore[index]
                    "object_disposition"
                ]
                for record_checkpoint, _, _ in records
            ),
            "canonical_bytes_verified": True,
            "content_addresses_verified": True,
            "ed25519_signatures_verified": True,
            "append_chain_verified": True,
            "if_absent_idempotent_complete_semantics_verified": True,
            "durability_verified": False,
            "live_cas_store_verified": False,
            "service_image_verified": False,
            "runtime_closure_verified": False,
            "live_uid_gid_verified": False,
            "live_control_domain_verified": False,
            "private_key_custody_verified": False,
            "external_anti_rollback_verified": False,
            "authority_evidence": False,
            "readiness_evidence": False,
            "execution_authorized": False,
            "claim_boundary": pure_claim_boundary,
            "rollback_disclosure": rollback_disclosure,
        }
        for name, value in values.items():
            object_setattr(result, name, value)
        register_result(result)
        return require_verified_cas_store_complete_session_replay_core_v2(
            result
        )

    def verify_cas_store_complete_session_replay_core_v2(
        *,
        stage_b_lock_payload: bytes,
        authority_profile_payload: bytes,
        signer_registry_payload: bytes,
        out_of_band_expected_lock_id_bytes: bytes,
        out_of_band_expected_lock_payload_sha256_digest: bytes,
        retention_plan_payload: bytes,
        request_payloads_in_exact_order: tuple[bytes, ...],
        object_payloads_in_exact_order: tuple[bytes, ...],
        signed_receipt_payloads_in_exact_order: tuple[bytes, ...],
        checkpoint_payloads_in_exact_order: tuple[bytes, ...],
        head_payloads_in_exact_order: tuple[bytes, ...],
        response_payloads_in_exact_order: tuple[bytes, ...],
    ) -> VerifiedCasStoreCompleteSessionReplayCoreV2:
        """Replay one raw complete store/2 session without side effects."""

        plan = parse_plan(retention_plan_payload)
        authority_binding = load_authority_binding(
            stage_b_lock_payload=stage_b_lock_payload,
            authority_profile_payload=authority_profile_payload,
            signer_registry_payload=signer_registry_payload,
            out_of_band_expected_lock_id_bytes=(
                out_of_band_expected_lock_id_bytes
            ),
            out_of_band_expected_lock_payload_sha256_digest=(
                out_of_band_expected_lock_payload_sha256_digest
            ),
        )
        if (
            plan["authority_profile_id"]
            != authority_binding.authority_profile_id
            or plan["signer_registry_id"]
            != authority_binding.signer_registry_id
        ):
            fail(
                fail_authority,
                "complete-session plan differs from authority graph",
            )
        sequences = (
            request_payloads_in_exact_order,
            object_payloads_in_exact_order,
            signed_receipt_payloads_in_exact_order,
            checkpoint_payloads_in_exact_order,
            head_payloads_in_exact_order,
            response_payloads_in_exact_order,
        )
        if exact_any(
            exact_type(sequence) is not exact_tuple
            for sequence in sequences
        ):
            fail(
                fail_input,
                "complete-session artifacts must be exact tuples of bytes",
            )
        expected = plan["expected_object_count"]
        if exact_type(expected) is not exact_int:
            fail(fail_schema, "plan expected count type differs")
        if exact_any(
            exact_len(sequence) != expected for sequence in sequences
        ):
            fail(
                fail_incomplete,
                "complete-session artifact cardinality differs from plan",
            )
        if exact_any(
            exact_type(payload) is not exact_bytes
            for sequence in sequences
            for payload in sequence
        ):
            fail(
                fail_input,
                "complete-session artifacts must contain exact bytes only",
            )

        previous = genesis_checkpoint_sha256
        request_ids: set[str] = exact_set()
        seen_object_ids: set[str] = exact_set()
        records: list[
            tuple[
                Mapping[str, object],
                Mapping[str, object],
                Mapping[str, object],
            ]
        ] = []
        for index in exact_range(expected):
            request_payload = request_payloads_in_exact_order[index]
            object_payload = object_payloads_in_exact_order[index]
            receipt_payload = signed_receipt_payloads_in_exact_order[index]
            checkpoint_payload = checkpoint_payloads_in_exact_order[index]
            head_payload = head_payloads_in_exact_order[index]
            response_payload = response_payloads_in_exact_order[index]

            request = parse_request(request_payload, plan)
            receipt = parse_signed_receipt(
                decode(receipt_payload, max_receipt, "signed receipt"),
                plan,
                authority_binding,
            )
            checkpoint = parse_checkpoint(
                checkpoint_payload,
                raw=True,
                plan=plan,
                authority_binding=authority_binding,
            )
            head = parse_head(
                head_payload,
                raw=True,
                plan=plan,
                authority_binding=authority_binding,
            )
            response = parse_response(
                response_payload,
                raw=True,
                plan=plan,
                authority_binding=authority_binding,
            )
            binding = verify_triplet(
                checkpoint=checkpoint,
                head=head,
                response=response,
            )
            ordinal = binding["ordinal"]
            request_id = binding["request_id"]
            if (
                exact_type(ordinal) is not exact_int
                or ordinal != index
                or binding["previous_checkpoint_sha256"] != previous
                or exact_type(request_id) is not exact_str
                or request_id in request_ids
            ):
                fail(
                    fail_chain,
                    "pure replay found a gap, fork, or duplicate request",
                )
            if (
                not exact_equal(request, transition_to_request(binding))
                or canonical(request) != request_payload
                or not exact_equal(receipt, checkpoint["signed_receipt"])
                or canonical(receipt) != receipt_payload
                or canonical(response["checkpoint_record"])
                != checkpoint_payload
                or canonical(response["head_record"]) != head_payload
            ):
                fail(
                    fail_binding,
                    "presented request/receipt/checkpoint/head bytes differ "
                    "from signed graph",
                )
            objects = plan["objects_in_exact_order"]
            if (
                exact_type(objects) is not exact_list
                or exact_type(objects[index]) is not exact_dict
            ):
                fail(fail_schema, "planned object sequence type differs")
            maximum_bytes = objects[index]["maximum_bytes"]
            if (
                exact_type(maximum_bytes) is not exact_int
                or not 0 < exact_len(object_payload) <= maximum_bytes
            ):
                fail(
                    fail_binding,
                    "presented object bytes are outside planned maximum",
                )
            object_sha = sha256_bytes(object_payload)
            if (
                object_sha != binding["object_id"]
                or object_sha != request["bound_artifact_sha256"]
                or exact_len(object_payload)
                != binding["object_size_bytes"]
                or exact_len(object_payload)
                != request["reported_size_bytes"]
            ):
                fail(
                    fail_binding,
                    "presented object bytes differ from signed request/object",
                )
            signed_body = receipt["signed_body"]
            if exact_type(signed_body) is not exact_dict:
                fail(fail_schema, "signed receipt body type differs")
            expected_disposition = (
                "PRESENT_VERIFIED"
                if object_sha in seen_object_ids
                else "CREATED_IF_ABSENT"
            )
            if (
                signed_body["object_disposition"]
                != expected_disposition
            ):
                fail(
                    fail_binding,
                    "signed disposition differs from exact IF_ABSENT order",
                )
            request_ids.add(request_id)
            seen_object_ids.add(object_sha)
            previous = sha256_bytes(canonical(checkpoint))
            records.append((checkpoint, head, response))

        if (
            not records
            or records[-1][1]["binding"]["complete"] is not True  # type: ignore[index]
            or records[-1][1]["binding"]["observed_object_count"]  # type: ignore[index]
            != expected
            or exact_any(
                record_head["binding"]["complete"] is True  # type: ignore[index]
                for _, record_head, _ in records[:-1]
            )
        ):
            fail(
                fail_incomplete,
                "presented session lacks one exact terminal complete head",
            )
        return issue_result(
            plan=plan,
            authority_binding=authority_binding,
            records=exact_tuple(records),
        )

    return (
        VerifiedCasStoreCompleteSessionReplayCoreV2,
        verify_cas_store_complete_session_replay_core_v2,
        require_verified_cas_store_complete_session_replay_core_v2,
    )


(
    VerifiedCasStoreCompleteSessionReplayCoreV2,
    verify_cas_store_complete_session_replay_core_v2,
    require_verified_cas_store_complete_session_replay_core_v2,
) = _build_store_replay_core_v2()
del _build_store_replay_core_v2


__all__ = [
    "SCHEMA_VERSION",
    "INSPECTED_STORE_V2_SOURCE_SHA256",
    "PURE_REPLAY_CLAIM_BOUNDARY",
    "FAIL_CLOSED_ROLLBACK_DISCLOSURE",
    "FAIL_INPUT",
    "FAIL_CANONICAL",
    "FAIL_SCHEMA",
    "FAIL_BINDING",
    "FAIL_AUTHORITY",
    "FAIL_SIGNATURE",
    "FAIL_CHAIN",
    "FAIL_INCOMPLETE",
    "CasStoreReplayCoreV2Error",
    "VerifiedCasStoreCompleteSessionReplayCoreV2",
    "verify_cas_store_complete_session_replay_core_v2",
    "require_verified_cas_store_complete_session_replay_core_v2",
]
