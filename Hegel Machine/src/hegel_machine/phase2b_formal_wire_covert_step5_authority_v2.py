"""Pure candidate authority/profile V2 for Phase-2B Step 5.

V2 freezes a closed Stage-B lock -> authority profile -> exact eight-role
registry graph and five typed *static* service bindings.  The services are
``runner``, ``pure_replayer``, ``cas_store``, ``logical_time``, and
``governance_status``.  A static binding records requested OCI and runtime-
closure policy identities only; it is not evidence that an image exists, was
executed, or had its live runtime closure observed.

The governance/source/pure-replayer purpose lists contain the purpose cascade
needed by later owner-status and R01 work.  This module verifies registry
signatures with the concrete ``cryptography`` Ed25519 backend, but it does not
issue keys, signatures, R01 receipts, readiness overlays, images, attempts, or
live evidence.  Every readiness node remains false.

Production loaders accept canonical bytes only.  The Stage-B loader additionally
requires an ASCII lock-ID byte string and a raw 32-byte SHA-256 digest supplied
out of band.  Decoded-object verification and injected verifier callbacks are
intentionally unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from types import FunctionType, MappingProxyType
from typing import Final, Mapping, NoReturn

from .rfc8785_v1 import (
    RFC8785Error,
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)

try:  # Importability remains; cryptographic verification still fails closed.
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - depends on the pinned verifier image.
    _InvalidSignature = None
    _Ed25519PublicKey = None


AUTHORITY_PROFILE_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-authority-profile/2"
)
STAGE_B_TRUST_LOCK_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-stage-b-trust-lock/2"
)
SIGNER_REGISTRY_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-signer-registry/2"
)
DETACHED_SIGNATURE_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-detached-signature/2"
)
SERVICE_STATIC_BINDING_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-service-static-binding/2"
)
READINESS_CANDIDATE_SNAPSHOT_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-readiness-candidate-snapshot/2"
)
ROLE_KEY_SCHEMA_V1: Final = "hegel-machine-phase2b-step5-role-key/1"

AUTHORITY_PROFILE_PREFIX_V2: Final = "phase2b_step5_authority_profile_v2_"
STAGE_B_TRUST_LOCK_PREFIX_V2: Final = "phase2b_step5_stage_b_trust_lock_v2_"
SIGNER_REGISTRY_PREFIX_V2: Final = "phase2b_step5_signer_registry_v2_"
SERVICE_STATIC_BINDING_PREFIX_V2: Final = (
    "phase2b_step5_service_static_binding_v2_"
)
READINESS_CANDIDATE_PREFIX_V2: Final = (
    "phase2b_step5_readiness_candidate_v2_"
)
# Role-key/1 is deliberately preserved so existing store/time/status actors
# compute the same content address from the new full purpose lists.
ROLE_KEY_PREFIX_V1: Final = "phase2b_step5_role_key_v1_"

OWNER_AUTHORIZATION_ID_V1: Final = (
    "phase2b_c1_owner_actual_unsealed_960_authorization_v1_"
    "dd2cd6a98b429fe01f172daf16ea70404c91356e40e9e42daed4cb5416a5c50d"
)
OWNER_AUTHORIZATION_CONFIG_SHA256_V1: Final = (
    "26ff84c3fe239158b1e8db24b5634ccfe8b3f3995d54bdc1f728c49a70f8fca1"
)
OWNER_AUTHORIZATION_SOURCE_SHA256_V1: Final = (
    "4c96d50706e520f1c67b2d2c31fce9ed605feacbf4c1177ddf9a7a2a9c1d9a73"
)
READINESS_DAG_ID_V1: Final = (
    "phase2b_actual_unsealed_960_readiness_dag_v1_"
    "775b86cb499db3a991686634335ec444f7e44c7caea92b1c2b948036b585ea84"
)
PROTOCOL_ID_V1: Final = (
    "phase2b_protocol_"
    "62ad411b5b5a0f912626c54e4bd822a8c585a8f612f5fce6040cce500a11756a"
)
EXACT_FREEZE_VERSION_V1: Final = "hegel-freeze-p2b-p3-v1.0.1"
EXACT_FREEZE_ID_V1: Final = (
    "phase2b_exact_freeze_"
    "ffa1fd4fed0b5c2c018803aa9f730b8c85c144efe7e4aa324256681d1c742cbe"
)
AUTHORITY_CLASS_V1: Final = "OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1"
AUTHORITY_REQUIREMENT_ID_V1: Final = (
    "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
    "53631d563f8da8f26c7202b7dd0681be5423fe2ca5c15e9a9f5deaa2920d9940"
)
ACTOR_PROFILE_CONTENT_ID_V1: Final = (
    "phase2b_c1_container_actor_profile_v2_"
    "b2f081771f3de8d2a531b08b58109d95c51815904e9612e66f1bbfe69babf906"
)
ACTOR_PROFILE_CANONICAL_SHA256_V1: Final = (
    "8ca1850f026dabb7a4b69add0f89b33a3fcfb789223b2fd95ef0379608adbeed"
)
LOGICAL_STAGE_WIRE_SCHEMA_ID_V1: Final = (
    "phase2b_c1_logical_stage_wire_schema_v1_"
    "1a625fbc369b057b9fc295fec02391d0421fc071ca41905b15901d4365833b5d"
)
LOGICAL_STAGE_WIRE_POLICY_ID_V1: Final = (
    "phase2b_c1_logical_stage_wire_policy_v2_"
    "c88653f2caa92eae1dfb7456217c4ca05c339e8f23e0540a06e435552dc59573"
)
STAGE_EVENT_CONTRACT_MANIFEST_ID_V1: Final = (
    "phase2b_c1_stage_event_contract_manifest_v1_"
    "c0929d05bdae56205e08700ab760e88a4668c686e73fe4be7a00ce6d69f9baf7"
)

QV2_PROFILE_ID_V1: Final = (
    "phase2b_formal_wire_covert_qualification_profile_v2_"
    "ad20443a8f68a9f07492a532adb389c8e73234d343a14bc547a91d868a935d6b"
)
QV2_PROFILE_PAYLOAD_SHA256_V1: Final = (
    "1c2f40f8227a8e3b7f7d56bce00db8ea410d60ac3dd006fd53d7fd7fa7b2297e"
)
QV2_SOURCE_SHA256_V1: Final = (
    "921bbb771d27c339acaf2c5e1e48f35a899c56552929065c02894e590d32e2ed"
)
QV2_DEPENDENCY_LOCK_SHA256_V1: Final = (
    "0ccd291a28df11fa0e7e7399a086bf01516a20fa2aa5e8abf3a05687b54b85db"
)
EXACT_QV2_BINDING_V1: Final = MappingProxyType(
    {
        "schema_version": "hegel-machine-phase2b-formal-wire-covert-qualification/9",
        "profile_id": QV2_PROFILE_ID_V1,
        "profile_payload_sha256": QV2_PROFILE_PAYLOAD_SHA256_V1,
        "source_sha256": QV2_SOURCE_SHA256_V1,
        "dependency_lock_sha256": QV2_DEPENDENCY_LOCK_SHA256_V1,
    }
)

ROLE_ORDER_V2: Final = (
    "GOVERNANCE_ANCHOR",
    "SOURCE_RUNTIME_CUSTODIAN",
    "WIRE_QUALIFIER_R03",
    "AUDITOR_QUALIFIER_R04",
    "INTEGRATION_QUALIFIER_R05",
    "CAS_STORE",
    "LOGICAL_TIME",
    "PURE_REPLAYER",
)
ROLE_CONTROL_DOMAINS_V2: Final = MappingProxyType(
    {
        "GOVERNANCE_ANCHOR": "STEP5_GOVERNANCE_CONTROL",
        "SOURCE_RUNTIME_CUSTODIAN": "STEP5_SOURCE_RUNTIME_CUSTODY_CONTROL",
        "WIRE_QUALIFIER_R03": "STEP5_R03_WIRE_QUALIFICATION_CONTROL",
        "AUDITOR_QUALIFIER_R04": "STEP5_R04_AUDITOR_QUALIFICATION_CONTROL",
        "INTEGRATION_QUALIFIER_R05": "STEP5_R05_INTEGRATION_QUALIFICATION_CONTROL",
        "CAS_STORE": "STEP5_CAS_STORE_CONTROL",
        "LOGICAL_TIME": "STEP5_LOGICAL_TIME_CONTROL",
        "PURE_REPLAYER": "STEP5_PURE_REPLAY_CONTROL",
    }
)
ROLE_SIGNATURE_PURPOSES_V2: Final = MappingProxyType(
    {
        "GOVERNANCE_ANCHOR": (
            "SIGN_SIGNER_REGISTRY",
            "SIGN_OWNER_AUTHORIZATION_STATUS_EVENT",
        ),
        "SOURCE_RUNTIME_CUSTODIAN": (
            "SIGN_R01_AUTHORIZATION_STATUS_CANDIDATE",
            "SIGN_R02_SYNTHETIC_BRANCH_CANDIDATE",
            "SIGN_RUNTIME_CLOSURE",
            "SIGN_R01_AUTHORIZATION_STATUS_RECEIPT",
        ),
        "WIRE_QUALIFIER_R03": ("SIGN_R03_WIRE_QUALIFICATION",),
        "AUDITOR_QUALIFIER_R04": ("SIGN_R04_AUDITOR_QUALIFICATION",),
        "INTEGRATION_QUALIFIER_R05": ("SIGN_R05_INTEGRATION_QUALIFICATION",),
        "CAS_STORE": ("SIGN_CAS_IF_ABSENT_RECEIPT",),
        "LOGICAL_TIME": ("SIGN_LOGICAL_TIME_TOKEN",),
        "PURE_REPLAYER": (
            "COUNTERSIGN_R01_AUTHORIZATION_STATUS_CANDIDATE",
            "COUNTERSIGN_R02_SYNTHETIC_BRANCH_CANDIDATE",
            "SIGN_PURE_REPLAY_RESULT",
            "COUNTERSIGN_R01_AUTHORIZATION_STATUS_RECEIPT",
            "SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR",
        ),
    }
)

SERVICE_ORDER_V2: Final = (
    "runner",
    "pure_replayer",
    "cas_store",
    "logical_time",
    "governance_status",
)
SERVICE_ROLE_V2: Final = MappingProxyType(
    {
        "runner": "SOURCE_RUNTIME_CUSTODIAN",
        "pure_replayer": "PURE_REPLAYER",
        "cas_store": "CAS_STORE",
        "logical_time": "LOGICAL_TIME",
        "governance_status": "GOVERNANCE_ANCHOR",
    }
)
SERVICE_UID_GID_V2: Final = MappingProxyType(
    {
        "runner": (41001, 42001),
        "pure_replayer": (41002, 42002),
        "cas_store": (41003, 42003),
        "logical_time": (41004, 42004),
        "governance_status": (41005, 42005),
    }
)

READINESS_NODE_ORDER_V1: Final = (
    "R01_AUTHORIZATION_VERIFIED",
    "R02_EXECUTION_BRANCH_BINDINGS_REPLAYED",
    "R03_FORMAL_TRUSTED_WIRE_IMPLEMENTATION_QUALIFIED",
    "R04_FORMAL_COVERT_AUDITOR_IMPLEMENTATION_QUALIFIED",
    "R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED",
    "R06_GENERATOR_IMPLEMENTATION_QUALIFIED",
    "R07_BASELINE_REGISTRATIONS_FROZEN",
    "R08_STATISTICAL_VALIDATION_PASSED",
    "R09_SIX_ROLE_LIVE_DOCKER_QUALIFIED",
    "R10_DURABLE_CUSTODY_AND_LEDGER_VERIFIED",
    "R11_EXECUTION_FREEZE_COMMITMENTS_BOUND",
    "R12_INDEPENDENT_LAUNCH_OVERLAY_VERIFIED",
    "R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY",
    "R14_ACTUAL_UNSEALED_960_COHORT_GENERATED",
    "R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED",
    "R16_INFERENCE_ADMISSION_UNLOCKED",
)
READINESS_DEPENDENCIES_V1: Final = (
    (),
    (READINESS_NODE_ORDER_V1[0],),
    (READINESS_NODE_ORDER_V1[1],),
    (READINESS_NODE_ORDER_V1[2],),
    (READINESS_NODE_ORDER_V1[3],),
    (READINESS_NODE_ORDER_V1[4],),
    (READINESS_NODE_ORDER_V1[4],),
    (READINESS_NODE_ORDER_V1[5], READINESS_NODE_ORDER_V1[6]),
    (READINESS_NODE_ORDER_V1[7],),
    (READINESS_NODE_ORDER_V1[8],),
    (READINESS_NODE_ORDER_V1[9],),
    (READINESS_NODE_ORDER_V1[10],),
    (READINESS_NODE_ORDER_V1[11],),
    (READINESS_NODE_ORDER_V1[12],),
    (READINESS_NODE_ORDER_V1[13],),
    (READINESS_NODE_ORDER_V1[14],),
)

FAIL_INPUT_V2: Final = "FAIL_STEP5_AUTHORITY_V2_INPUT"
FAIL_CANONICAL_V2: Final = "FAIL_STEP5_AUTHORITY_V2_CANONICAL"
FAIL_SCHEMA_V2: Final = "FAIL_STEP5_AUTHORITY_V2_SCHEMA"
FAIL_CONTENT_ID_V2: Final = "FAIL_STEP5_AUTHORITY_V2_CONTENT_ID"
FAIL_BINDING_V2: Final = "FAIL_STEP5_AUTHORITY_V2_BINDING"
FAIL_SIGNATURE_BACKEND_V2: Final = "FAIL_STEP5_AUTHORITY_V2_SIGNATURE_BACKEND"
FAIL_SIGNATURE_V2: Final = "FAIL_STEP5_AUTHORITY_V2_SIGNATURE"
FAIL_CLAIM_BOUNDARY_V2: Final = "FAIL_STEP5_AUTHORITY_V2_CLAIM_BOUNDARY"
FAIL_TRUST_ANCHOR_V2: Final = "FAIL_STEP5_AUTHORITY_V2_TRUST_ANCHOR"
FAIL_NOT_IMPLEMENTED_V2: Final = "FAIL_STEP5_AUTHORITY_V2_NOT_IMPLEMENTED"

_MAX_CANONICAL_BYTES: Final = 8 * 1024 * 1024
_MAX_DEPTH: Final = 64
_MAX_NODES: Final = 250_000
_MAX_STRING_UTF8_BYTES: Final = 1 * 1024 * 1024
_SAFE_INTEGER: Final = (1 << 53) - 1
_PUBLIC_KEY_HEX_LENGTH: Final = 64
_SIGNATURE_HEX_LENGTH: Final = 128
_MAX_LAYERS: Final = 128
_HEX64_RE: Final = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}")
_OCI_DIGEST_RE: Final = re.compile(r"sha256:[0-9a-f]{64}")
_ID_RE: Final = re.compile(r"[a-z0-9][a-z0-9_./:-]{0,255}_[0-9a-f]{64}")

_DOMAINS: Final = MappingProxyType(
    {
        "stage_b_trust_lock": b"HEGEL/PHASE2B/STEP5/STAGE_B_TRUST_LOCK/V2\x00",
        "authority_profile": b"HEGEL/PHASE2B/STEP5/AUTHORITY_PROFILE/V2\x00",
        "signer_registry": b"HEGEL/PHASE2B/STEP5/SIGNER_REGISTRY/V2\x00",
        "role_key": b"HEGEL/PHASE2B/STEP5/ROLE_KEY/V1\x00",
        "service_static_binding": b"HEGEL/PHASE2B/STEP5/SERVICE_STATIC_BINDING/V2\x00",
        "readiness_candidate": b"HEGEL/PHASE2B/STEP5/READINESS_CANDIDATE/V2\x00",
    }
)
_PREFIXES: Final = MappingProxyType(
    {
        "stage_b_trust_lock": STAGE_B_TRUST_LOCK_PREFIX_V2,
        "authority_profile": AUTHORITY_PROFILE_PREFIX_V2,
        "signer_registry": SIGNER_REGISTRY_PREFIX_V2,
        "role_key": ROLE_KEY_PREFIX_V1,
        "service_static_binding": SERVICE_STATIC_BINDING_PREFIX_V2,
        "readiness_candidate": READINESS_CANDIDATE_PREFIX_V2,
    }
)

# Only domains used by this module or explicitly reserved by the new purpose
# cascade are listed.  CAS-store and logical-time actors keep their own frozen
# preimage formats and are not redefined here.
_SIGNATURE_DOMAINS: Final = MappingProxyType(
    {
        "SIGN_SIGNER_REGISTRY": b"HEGEL/PHASE2B/STEP5/SIGN/REGISTRY/V2\x00",
        "SIGN_OWNER_AUTHORIZATION_STATUS_EVENT": b"HEGEL/PHASE2B/STEP5/SIGN/OWNER_AUTHORIZATION_STATUS_EVENT/GOVERNANCE/V1\x00",
        "SIGN_R01_AUTHORIZATION_STATUS_CANDIDATE": b"HEGEL/PHASE2B/STEP5/SIGN/R01_CANDIDATE/CUSTODIAN/V1\x00",
        "SIGN_R02_SYNTHETIC_BRANCH_CANDIDATE": b"HEGEL/PHASE2B/STEP5/SIGN/R02_CANDIDATE/CUSTODIAN/V1\x00",
        "SIGN_RUNTIME_CLOSURE": b"HEGEL/PHASE2B/STEP5/SIGN/RUNTIME_CLOSURE/CUSTODIAN/V1\x00",
        "SIGN_R01_AUTHORIZATION_STATUS_RECEIPT": b"HEGEL/PHASE2B/STEP5/SIGN/R01_STATUS_RECEIPT/CUSTODIAN/V1\x00",
        "COUNTERSIGN_R01_AUTHORIZATION_STATUS_CANDIDATE": b"HEGEL/PHASE2B/STEP5/SIGN/R01_CANDIDATE/PURE/V1\x00",
        "COUNTERSIGN_R02_SYNTHETIC_BRANCH_CANDIDATE": b"HEGEL/PHASE2B/STEP5/SIGN/R02_CANDIDATE/PURE/V1\x00",
        "SIGN_PURE_REPLAY_RESULT": b"HEGEL/PHASE2B/STEP5/SIGN/PURE_REPLAY_RESULT/V1\x00",
        "COUNTERSIGN_R01_AUTHORIZATION_STATUS_RECEIPT": b"HEGEL/PHASE2B/STEP5/SIGN/R01_STATUS_RECEIPT/PURE/V1\x00",
        "SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR": b"HEGEL/PHASE2B/STEP5/SIGN/OWNER_STATUS_ANTI_FORK_ANCHOR/PURE/V1\x00",
        "SIGN_R03_WIRE_QUALIFICATION": b"HEGEL/PHASE2B/STEP5/SIGN/R03_WIRE_QUALIFICATION/V1\x00",
        "SIGN_R04_AUDITOR_QUALIFICATION": b"HEGEL/PHASE2B/STEP5/SIGN/R04_AUDITOR_QUALIFICATION/V1\x00",
        "SIGN_R05_INTEGRATION_QUALIFICATION": b"HEGEL/PHASE2B/STEP5/SIGN/R05_INTEGRATION_QUALIFICATION/V1\x00",
    }
)


class Phase2BStep5AuthorityErrorV2(ValueError):
    """Stable fail-closed error for candidate authority V2."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BStep5AuthorityErrorV2(code, detail)


def _snapshot(
    value: object,
    *,
    depth: int = 0,
    counter: list[int] | None = None,
    _max_nodes: int = _MAX_NODES,
    _max_depth: int = _MAX_DEPTH,
    _safe_integer: int = _SAFE_INTEGER,
    _max_string_bytes: int = _MAX_STRING_UTF8_BYTES,
    _fail_exact=_fail,
) -> object:
    active_counter = [0] if counter is None else counter

    def visit(item: object, item_depth: int) -> object:
        active_counter[0] += 1
        if active_counter[0] > _max_nodes or item_depth > _max_depth:
            _fail_exact(FAIL_CANONICAL_V2, "JSON structure exceeds frozen bounds")
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            if not -_safe_integer <= item <= _safe_integer:
                _fail_exact(
                    FAIL_CANONICAL_V2, "integer outside binary64-safe range"
                )
            return item
        if type(item) is float:
            _fail_exact(
                FAIL_CANONICAL_V2,
                "floating-point values are outside V2 schemas",
            )
        if type(item) is str:
            try:
                encoded = item.encode("utf-8")
            except UnicodeError as exc:
                raise Phase2BStep5AuthorityErrorV2(
                    FAIL_CANONICAL_V2, "invalid Unicode"
                ) from exc
            if len(encoded) > _max_string_bytes:
                _fail_exact(FAIL_CANONICAL_V2, "string exceeds UTF-8 bound")
            return item
        if type(item) is list:
            return [visit(child, item_depth + 1) for child in item]
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                _fail_exact(
                    FAIL_CANONICAL_V2, "object keys must be exact strings"
                )
            return {
                key: visit(child, item_depth + 1)
                for key, child in item.items()
            }
        _fail_exact(FAIL_CANONICAL_V2, "value is outside exact JSON data model")

    return visit(value, depth)


def _restore_integral_numbers(
    value: object, _safe_integer: int = _SAFE_INTEGER
) -> object:
    def restore(item: object) -> object:
        if (
            type(item) is float
            and item.is_integer()
            and abs(item) <= _safe_integer
        ):
            return int(item)
        if type(item) is list:
            return [restore(child) for child in item]
        if type(item) is dict:
            return {key: restore(child) for key, child in item.items()}
        return item

    return restore(value)


def canonical_step5_json_bytes_v2(
    value: object,
    _snapshot_exact=_snapshot,
    _canonicalize=canonicalize_rfc8785_v1,
    _max_canonical_bytes: int = _MAX_CANONICAL_BYTES,
    _fail_exact=_fail,
) -> bytes:
    """Return bounded exact RFC-8785 bytes without transport newline."""

    try:
        encoded = _canonicalize(_snapshot_exact(value))  # type: ignore[arg-type]
    except (RFC8785Error, TypeError, ValueError, RecursionError) as exc:
        raise Phase2BStep5AuthorityErrorV2(FAIL_CANONICAL_V2, str(exc)) from exc
    if (
        not 1 <= len(encoded) <= _max_canonical_bytes
        or encoded.endswith(b"\n")
    ):
        _fail_exact(FAIL_CANONICAL_V2, "canonical byte size/newline boundary differs")
    return encoded


def decode_canonical_step5_json_v2(
    payload: bytes,
    *,
    maximum_bytes: int = _MAX_CANONICAL_BYTES,
    _decode_exact=decode_and_require_canonical_rfc8785_v1,
    _restore_exact=_restore_integral_numbers,
    _snapshot_exact=_snapshot,
    _canonical=canonical_step5_json_bytes_v2,
    _max_canonical_bytes: int = _MAX_CANONICAL_BYTES,
    _fail_exact=_fail,
) -> object:
    """Decode byte-for-byte canonical RFC-8785 under structural bounds."""

    if (
        type(payload) is not bytes
        or type(maximum_bytes) is not int
        or not 1 <= maximum_bytes <= _max_canonical_bytes
        or not 1 <= len(payload) <= maximum_bytes
        or payload.endswith(b"\n")
    ):
        _fail_exact(
            FAIL_INPUT_V2,
            "payload must be bounded exact bytes without terminal LF",
        )
    try:
        decoded = _decode_exact(payload)
    except (RFC8785Error, TypeError, ValueError, RecursionError) as exc:
        raise Phase2BStep5AuthorityErrorV2(FAIL_CANONICAL_V2, str(exc)) from exc
    restored = _snapshot_exact(_restore_exact(decoded))
    if _canonical(restored) != payload:
        _fail_exact(FAIL_CANONICAL_V2, "canonical replay differs")
    return restored


def phase2b_step5_content_id_v2(
    kind: str,
    body: object,
    _domains=_DOMAINS,
    _prefixes=_PREFIXES,
    _sha256=hashlib.sha256,
    _canonical=canonical_step5_json_bytes_v2,
    _fail_exact=_fail,
) -> str:
    if type(kind) is not str or kind not in _domains:
        _fail_exact(FAIL_INPUT_V2, "unknown V2 content-ID kind")
    return _prefixes[kind] + _sha256(
        _domains[kind] + _canonical(body)
    ).hexdigest()


def _object(value: object, fields: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(fields):
        _fail(FAIL_SCHEMA_V2, f"{name} exact field set differs")
    return value


def _text(value: object, name: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > maximum:
        _fail(FAIL_SCHEMA_V2, f"{name} must be bounded non-empty string")
    return value


def _sha(
    value: object,
    name: str,
    _text_exact=_text,
    _hex64_re=_HEX64_RE,
    _fail_exact=_fail,
) -> str:
    text = _text_exact(value, name)
    if _hex64_re.fullmatch(text) is None or text == "0" * 64:
        _fail_exact(FAIL_SCHEMA_V2, f"{name} must be nonzero lowercase SHA-256")
    return text


def _identifier(
    value: object,
    prefix: str,
    name: str,
    _text_exact=_text,
    _hex64_re=_HEX64_RE,
    _fail_exact=_fail,
) -> str:
    text = _text_exact(value, name)
    if (
        not text.startswith(prefix)
        or _hex64_re.fullmatch(text[len(prefix) :]) is None
    ):
        _fail_exact(FAIL_SCHEMA_V2, f"{name} content-ID syntax differs")
    return text


def _generic_identifier(
    value: object,
    name: str,
    _text_exact=_text,
    _id_re=_ID_RE,
    _fail_exact=_fail,
) -> str:
    text = _text_exact(value, name)
    if _id_re.fullmatch(text) is None:
        _fail_exact(FAIL_SCHEMA_V2, f"{name} identifier syntax differs")
    return text


def _oci_digest(
    value: object,
    name: str,
    _text_exact=_text,
    _oci_digest_re=_OCI_DIGEST_RE,
    _fail_exact=_fail,
) -> str:
    text = _text_exact(value, name)
    if (
        _oci_digest_re.fullmatch(text) is None
        or text == "sha256:" + "0" * 64
    ):
        _fail_exact(FAIL_SCHEMA_V2, f"{name} OCI digest syntax differs")
    return text


def _exact_integer(value: object, expected: int, name: str) -> int:
    if type(value) is not int or value != expected:
        _fail(FAIL_SCHEMA_V2, f"{name} must be exact integer {expected}")
    return value


def _exact_equal(
    left: object, right: object, _mapping_type=Mapping
) -> bool:
    def equal(item: object, other: object) -> bool:
        if isinstance(item, _mapping_type) and isinstance(other, _mapping_type):
            return set(item) == set(other) and all(
                equal(item[key], other[key]) for key in item
            )
        if isinstance(item, (list, tuple)) and isinstance(other, (list, tuple)):
            return len(item) == len(other) and all(
                equal(child, counterpart)
                for child, counterpart in zip(item, other, strict=True)
            )
        return type(item) is type(other) and item == other

    return equal(left, right)


def _freeze(value: object, _mapping_proxy=MappingProxyType) -> object:
    def freeze(item: object) -> object:
        if type(item) is dict:
            return _mapping_proxy(
                {key: freeze(child) for key, child in item.items()}
            )
        if type(item) is list:
            return tuple(freeze(child) for child in item)
        return item

    return freeze(value)


@dataclass(frozen=True, slots=True, init=False)
class Phase2BStep5VerifiedArtifactV2:
    kind: str
    artifact_id: str
    body: Mapping[str, object]
    signature_backends: tuple[str, ...]

    def __new__(
        cls, *args: object, **kwargs: object
    ) -> "Phase2BStep5VerifiedArtifactV2":
        raise TypeError("verified Step-5 V2 artifacts are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("verified Step-5 V2 artifacts are privately issued")


def _build_private_verified_artifact_lineage_v2():
    """Create a closed, mutation-detecting artifact lineage.

    No issuance token or issuer is retained as a module global after the public
    byte loaders have captured the two closure functions below.  The closure
    also keeps a strong identity record for every issued object, so
    ``object.__new__`` construction, copying a field set, or mutating a genuine
    frozen dataclass via ``object.__setattr__`` cannot cross the consumer
    boundary.  This is an ordinary Python caller/KAT boundary, not resistance
    to arbitrary interpreter-memory introspection or code mutation.
    """

    result_type = Phase2BStep5VerifiedArtifactV2
    snapshot_exact = _snapshot
    freeze_exact = _freeze
    content_id_exact = phase2b_step5_content_id_v2
    fail_exact = _fail
    allowed_kinds = frozenset(
        {
            "stage_b_trust_lock",
            "authority_profile",
            "signer_registry",
            "readiness_candidate",
        }
    )
    issued: dict[
        int,
        tuple[
            Phase2BStep5VerifiedArtifactV2,
            str,
            str,
            Mapping[str, object],
            tuple[str, ...],
        ],
    ] = {}

    def require(
        value: object, kind: str, name: str
    ) -> Phase2BStep5VerifiedArtifactV2:
        if type(value) is not result_type or type(kind) is not str:
            fail_exact(FAIL_INPUT_V2, f"verified {name} required")
        record = issued.get(id(value))
        if record is None or record[0] is not value:
            fail_exact(
                FAIL_INPUT_V2,
                f"verified {name} lacks the private issuance lineage",
            )
        _, expected_kind, expected_id, expected_body, expected_backends = record
        try:
            intact = (
                value.kind == expected_kind == kind
                and type(value.kind) is str
                and value.artifact_id == expected_id
                and type(value.artifact_id) is str
                and value.body is expected_body
                and value.signature_backends is expected_backends
            )
        except (AttributeError, TypeError):
            intact = False
        if not intact:
            fail_exact(
                FAIL_INPUT_V2,
                f"verified {name} private issuance lineage was mutated",
            )
        return value

    def issue(
        kind: str,
        artifact_id: str,
        body: dict[str, object],
        backends: tuple[str, ...] = (),
    ) -> Phase2BStep5VerifiedArtifactV2:
        if (
            type(kind) is not str
            or kind not in allowed_kinds
            or type(artifact_id) is not str
            or type(body) is not dict
            or type(backends) is not tuple
            or any(type(backend) is not str for backend in backends)
        ):
            fail_exact(FAIL_INPUT_V2, "private artifact issuance input differs")
        if artifact_id != content_id_exact(kind, body):
            fail_exact(
                FAIL_CONTENT_ID_V2,
                "private artifact issuance content address differs",
            )
        if (kind == "signer_registry") is not (
            backends == ("CRYPTOGRAPHY_ED25519",)
        ):
            fail_exact(
                FAIL_SIGNATURE_V2,
                "private artifact issuance signature lineage differs",
            )
        frozen = freeze_exact(snapshot_exact(body))
        assert isinstance(frozen, Mapping)
        result = object.__new__(result_type)
        object.__setattr__(result, "kind", kind)
        object.__setattr__(result, "artifact_id", artifact_id)
        object.__setattr__(result, "body", frozen)
        object.__setattr__(result, "signature_backends", backends)
        issued[id(result)] = (result, kind, artifact_id, frozen, backends)
        return require(result, kind, kind)

    return issue, require


(
    _hidden_issue_verified_artifact_v2,
    _hidden_require_verified_artifact_v2,
) = _build_private_verified_artifact_lineage_v2()
del _build_private_verified_artifact_lineage_v2


ROLE_KEY_FIELDS_V2: Final = frozenset(
    {
        "schema_version",
        "role",
        "control_domain",
        "authorized_signature_purposes",
        "algorithm",
        "key_epoch",
        "public_key_hex",
        "key_id",
    }
)
SERVICE_STATIC_BINDING_FIELDS_V2: Final = frozenset(
    {
        "schema_version",
        "service_name",
        "service_type",
        "binding_status",
        "role",
        "control_domain",
        "uid",
        "gid",
        "role_key_id",
        "key_epoch",
        "authorized_signature_purposes",
        "oci_index_digest",
        "oci_manifest_digest",
        "oci_config_digest",
        "ordered_layer_digests",
        "static_runtime_closure_policy_sha256",
        "binding_id",
    }
)
STAGE_B_TRUST_LOCK_BODY_FIELDS_V2: Final = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "canonicalization",
        "authority_profile_schema",
        "authority_profile_id",
        "authority_profile_payload_sha256",
        "signer_registry_schema",
        "signer_registry_id",
        "signer_registry_payload_sha256",
        "qv2_binding",
    }
)
AUTHORITY_PROFILE_BODY_FIELDS_V2: Final = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "canonicalization",
        "profile_epoch",
        "owner_authorization_binding",
        "readiness_dag_binding",
        "readiness_state",
        "qv2_binding",
        "source_binding",
        "service_static_bindings_in_exact_order",
        "role_policy",
        "governance_root",
        "threat_model",
        "permanent_nonclaims",
    }
)
SIGNER_REGISTRY_BODY_FIELDS_V2: Final = frozenset(
    {
        "schema_version",
        "authority_profile_id",
        "registry_epoch",
        "status",
        "claim_level",
        "revocation_epoch",
        "role_keys_in_exact_order",
        "readiness_state",
    }
)
DETACHED_SIGNATURE_FIELDS_V2: Final = frozenset(
    {
        "schema_version",
        "algorithm",
        "role",
        "purpose",
        "key_epoch",
        "signer_key_id",
        "signature_hex",
    }
)

SERVICE_TYPES_V2: Final = MappingProxyType(
    {
        "runner": "RUNNER_STATIC_SERVICE",
        "pure_replayer": "PURE_REPLAYER_STATIC_SERVICE",
        "cas_store": "CAS_STORE_STATIC_SERVICE",
        "logical_time": "LOGICAL_TIME_STATIC_SERVICE",
        "governance_status": "GOVERNANCE_STATUS_STATIC_SERVICE",
    }
)

_OWNER_BINDING_FIELDS_V2: Final = frozenset(
    {
        "authorization_schema",
        "authorization_id",
        "authorization_config_sha256",
        "authorization_source_sha256",
        "protocol_id",
        "exact_freeze_version",
        "exact_freeze_id",
        "authority_class",
        "authority_requirement_id",
        "actor_profile_content_id",
        "actor_profile_canonical_sha256",
        "logical_stage_wire_schema_id",
        "logical_stage_wire_policy_id",
        "stage_event_contract_manifest_id",
    }
)
_EXACT_OWNER_BINDING_V2: Final = MappingProxyType(
    {
        "authorization_schema": (
            "hegel-machine-phase2b-c1-owner-actual-unsealed-960-authorization/1"
        ),
        "authorization_id": OWNER_AUTHORIZATION_ID_V1,
        "authorization_config_sha256": OWNER_AUTHORIZATION_CONFIG_SHA256_V1,
        "authorization_source_sha256": OWNER_AUTHORIZATION_SOURCE_SHA256_V1,
        "protocol_id": PROTOCOL_ID_V1,
        "exact_freeze_version": EXACT_FREEZE_VERSION_V1,
        "exact_freeze_id": EXACT_FREEZE_ID_V1,
        "authority_class": AUTHORITY_CLASS_V1,
        "authority_requirement_id": AUTHORITY_REQUIREMENT_ID_V1,
        "actor_profile_content_id": ACTOR_PROFILE_CONTENT_ID_V1,
        "actor_profile_canonical_sha256": ACTOR_PROFILE_CANONICAL_SHA256_V1,
        "logical_stage_wire_schema_id": LOGICAL_STAGE_WIRE_SCHEMA_ID_V1,
        "logical_stage_wire_policy_id": LOGICAL_STAGE_WIRE_POLICY_ID_V1,
        "stage_event_contract_manifest_id": STAGE_EVENT_CONTRACT_MANIFEST_ID_V1,
    }
)

_THREAT_MODEL_V2: Final = MappingProxyType(
    {
        "same_admin_controller": True,
        "owner_accepted_technical_actor_model": True,
        "producer_may_be_malicious": True,
        "producer_outside_authority_roles": True,
        "producer_local_filesystem_claims_authoritative": False,
        "producer_local_filesystem_is_non_authoritative": True,
        "purpose_separated_keys_required": True,
        "distinct_control_domains_required": True,
        "static_distinct_service_uids_gids_requested": True,
        "static_oci_graphs_requested": True,
        "service_image_existence_verified": False,
        "service_image_pull_or_execution_verified": False,
        "live_runtime_closure_verified": False,
        "live_distinct_service_uids_gids_verified": False,
        "organizational_independence": False,
        "independent_human_actors": False,
        "external_authority": False,
        "remote_attestation": False,
        "hardware_key_nonexportability": False,
        "malicious_admin_resistance": False,
        "malicious_host_root_resistance": False,
        "malicious_container_runtime_resistance": False,
    }
)
_PERMANENT_NONCLAIMS_V2: Final = MappingProxyType(
    {
        "actual_corpus_input": False,
        "actual_campaign_id_allocated": False,
        "actual_attempt_id_allocated": False,
        "actual_execution_authorized": False,
        "r01_satisfied": False,
        "r02_satisfied": False,
        "r03_satisfied": False,
        "r04_satisfied": False,
        "r05_satisfied": False,
        "r06_or_later_unlocked": False,
        "service_images_exist": False,
        "service_images_executed": False,
        "service_static_bindings_live_verified": False,
        "runtime_closure_observed": False,
        "cas_store_verified": False,
        "logical_time_verified": False,
        "governance_status_ledger_admitted": False,
        "latest_owner_authorization_status_verified": False,
        "owner_status_external_anti_fork_anchor_verified": False,
        "r01_authorization_status_receipt_issued": False,
        "readiness_evidence_overlay_issued": False,
        "technical_control_domains_live_verified": False,
        "organizational_independence": False,
        "independent_human_actors": False,
        "external_authority": False,
        "remote_attestation": False,
        "hardware_key_nonexportability": False,
        "malicious_admin_resistance": False,
    }
)


def _exact_boolean(value: object, expected: bool, name: str) -> bool:
    if type(value) is not bool or value is not expected:
        _fail(FAIL_SCHEMA_V2, f"{name} must be exact boolean {expected}")
    return value


def _role_key_subject(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": row["schema_version"],
        "role": row["role"],
        "control_domain": row["control_domain"],
        "authorized_signature_purposes": row[
            "authorized_signature_purposes"
        ],
        "algorithm": row["algorithm"],
        "key_epoch": row["key_epoch"],
        "public_key_hex": row["public_key_hex"],
    }


def phase2b_step5_role_key_id_v2(
    row_without_or_with_id: object,
    _object_exact=_object,
    _content_id=phase2b_step5_content_id_v2,
    _subject=_role_key_subject,
    _fields=ROLE_KEY_FIELDS_V2,
) -> str:
    """Derive role-key/1 ID from the complete V2 role-purpose list."""

    fields = _fields
    if type(row_without_or_with_id) is dict and "key_id" not in row_without_or_with_id:
        fields = frozenset(_fields - {"key_id"})
    row = _object_exact(row_without_or_with_id, fields, "role key")
    return _content_id("role_key", _subject(row))


def _validate_role_key(
    row: object,
    expected_role: str,
    _object_exact=_object,
    _control_domains=ROLE_CONTROL_DOMAINS_V2,
    _purposes=ROLE_SIGNATURE_PURPOSES_V2,
    _text_exact=_text,
    _derive_key_id=phase2b_step5_role_key_id_v2,
    _fields=ROLE_KEY_FIELDS_V2,
    _role_key_schema: str = ROLE_KEY_SCHEMA_V1,
    _public_key_hex_length: int = _PUBLIC_KEY_HEX_LENGTH,
    _hex64_re=_HEX64_RE,
    _fail_exact=_fail,
) -> dict[str, object]:
    key = _object_exact(row, _fields, f"{expected_role} role key")
    if (
        key["schema_version"] != _role_key_schema
        or key["role"] != expected_role
        or key["control_domain"] != _control_domains[expected_role]
        or key["authorized_signature_purposes"]
        != list(_purposes[expected_role])
        or key["algorithm"] != "Ed25519"
        or type(key["key_epoch"]) is not int
        or key["key_epoch"] != 1
    ):
        _fail_exact(FAIL_SCHEMA_V2, f"{expected_role} complete key policy differs")
    public_key = _text_exact(key["public_key_hex"], f"{expected_role} public key")
    if (
        len(public_key) != _public_key_hex_length
        or _hex64_re.fullmatch(public_key) is None
        or public_key == "0" * _public_key_hex_length
    ):
        _fail_exact(FAIL_SCHEMA_V2, f"{expected_role} public key encoding differs")
    expected_id = _derive_key_id(key)
    if key["key_id"] != expected_id:
        _fail_exact(
            FAIL_CONTENT_ID_V2,
            f"{expected_role} key ID does not derive from its full purpose list",
        )
    return key


def _service_binding_subject(
    binding: Mapping[str, object],
    _fields=SERVICE_STATIC_BINDING_FIELDS_V2,
) -> dict[str, object]:
    return {
        key: binding[key]
        for key in _fields
        if key != "binding_id"
    }


def phase2b_step5_service_static_binding_id_v2(
    binding_without_or_with_id: object,
    _object_exact=_object,
    _content_id=phase2b_step5_content_id_v2,
    _subject=_service_binding_subject,
    _fields=SERVICE_STATIC_BINDING_FIELDS_V2,
) -> str:
    fields = _fields
    if (
        type(binding_without_or_with_id) is dict
        and "binding_id" not in binding_without_or_with_id
    ):
        fields = frozenset(_fields - {"binding_id"})
    binding = _object_exact(binding_without_or_with_id, fields, "service static binding")
    return _content_id(
        "service_static_binding", _subject(binding)
    )


def _validate_service_static_binding(
    value: object,
    expected_service: str,
    _object_exact=_object,
    _service_roles=SERVICE_ROLE_V2,
    _service_ids=SERVICE_UID_GID_V2,
    _service_types=SERVICE_TYPES_V2,
    _control_domains=ROLE_CONTROL_DOMAINS_V2,
    _purposes=ROLE_SIGNATURE_PURPOSES_V2,
    _identifier_exact=_identifier,
    _oci_digest_exact=_oci_digest,
    _sha_exact=_sha,
    _derive_binding_id=phase2b_step5_service_static_binding_id_v2,
    _fields=SERVICE_STATIC_BINDING_FIELDS_V2,
    _service_binding_schema: str = SERVICE_STATIC_BINDING_SCHEMA_V2,
    _role_key_prefix: str = ROLE_KEY_PREFIX_V1,
    _max_layers: int = _MAX_LAYERS,
    _fail_exact=_fail,
) -> dict[str, object]:
    binding = _object_exact(
        value,
        _fields,
        f"{expected_service} service static binding",
    )
    expected_role = _service_roles[expected_service]
    expected_uid, expected_gid = _service_ids[expected_service]
    if (
        binding["schema_version"] != _service_binding_schema
        or binding["service_name"] != expected_service
        or binding["service_type"] != _service_types[expected_service]
        or binding["binding_status"]
        != "STATIC_REQUESTED_BINDING_NOT_OBSERVED_LIVE"
        or binding["role"] != expected_role
        or binding["control_domain"]
        != _control_domains[expected_role]
        or type(binding["uid"]) is not int
        or binding["uid"] != expected_uid
        or type(binding["gid"]) is not int
        or binding["gid"] != expected_gid
        or type(binding["key_epoch"]) is not int
        or binding["key_epoch"] != 1
        or binding["authorized_signature_purposes"]
        != list(_purposes[expected_role])
    ):
        _fail_exact(FAIL_SCHEMA_V2, f"{expected_service} static service policy differs")
    _identifier_exact(
        binding["role_key_id"],
        _role_key_prefix,
        f"{expected_service} role-key ID",
    )
    image_graph = (
        _oci_digest_exact(binding["oci_index_digest"], f"{expected_service} OCI index"),
        _oci_digest_exact(
            binding["oci_manifest_digest"], f"{expected_service} OCI manifest"
        ),
        _oci_digest_exact(binding["oci_config_digest"], f"{expected_service} OCI config"),
    )
    if len(set(image_graph)) != 3:
        _fail_exact(FAIL_SCHEMA_V2, f"{expected_service} OCI graph digests collide")
    layers = binding["ordered_layer_digests"]
    if type(layers) is not list or not 1 <= len(layers) <= _max_layers:
        _fail_exact(
            FAIL_SCHEMA_V2,
            f"{expected_service} ordered OCI layers cardinality differs",
        )
    checked_layers = [
        _oci_digest_exact(item, f"{expected_service} OCI layer {index}")
        for index, item in enumerate(layers)
    ]
    if len(set(checked_layers)) != len(checked_layers):
        _fail_exact(FAIL_SCHEMA_V2, f"{expected_service} OCI layer list has duplicates")
    _sha_exact(
        binding["static_runtime_closure_policy_sha256"],
        f"{expected_service} static runtime-closure policy SHA",
    )
    expected_id = _derive_binding_id(binding)
    if binding["binding_id"] != expected_id:
        _fail_exact(FAIL_CONTENT_ID_V2, f"{expected_service} binding ID differs")
    return binding


def _validate_service_static_bindings(
    value: object,
    _services=SERVICE_ORDER_V2,
    _validate_binding=_validate_service_static_binding,
    _fail_exact=_fail,
) -> list[dict[str, object]]:
    if type(value) is not list or len(value) != len(_services):
        _fail_exact(FAIL_SCHEMA_V2, "exactly five ordered service bindings are required")
    bindings = [
        _validate_binding(row, service)
        for service, row in zip(_services, value, strict=True)
    ]
    for field in ("binding_id", "role_key_id", "uid", "gid"):
        if len({binding[field] for binding in bindings}) != len(bindings):
            _fail_exact(FAIL_SCHEMA_V2, f"service {field} values are not pairwise distinct")
    all_unix_ids = [binding["uid"] for binding in bindings] + [
        binding["gid"] for binding in bindings
    ]
    if len(set(all_unix_ids)) != 2 * len(bindings):
        _fail_exact(FAIL_SCHEMA_V2, "service UID/GID values are not globally distinct")
    all_oci_graph_digests = [
        binding[field]
        for binding in bindings
        for field in (
            "oci_index_digest",
            "oci_manifest_digest",
            "oci_config_digest",
        )
    ]
    if len(set(all_oci_graph_digests)) != len(all_oci_graph_digests):
        _fail_exact(FAIL_SCHEMA_V2, "service OCI index/manifest/config digests collide")
    return bindings


def _validate_owner_binding(
    value: object,
    _object_exact=_object,
    _equal_exact=_exact_equal,
    _fields=_OWNER_BINDING_FIELDS_V2,
    _expected=_EXACT_OWNER_BINDING_V2,
    _fail_exact=_fail,
) -> dict[str, object]:
    binding = _object_exact(
        value, _fields, "owner authorization binding"
    )
    if not _equal_exact(binding, _expected):
        _fail_exact(
            FAIL_BINDING_V2,
            "owner authorization binding differs from the frozen exact contract",
        )
    return binding


def _all_false_readiness_state_v2(
    _nodes=READINESS_NODE_ORDER_V1,
) -> dict[str, object]:
    return {
        "node_states_in_canonical_topological_order": [
            {"node_id": node, "satisfied": False}
            for node in _nodes
        ],
        "satisfied_node_count": 0,
        "highest_contiguous_satisfied_node": None,
        "all_nodes_satisfied": False,
        "readiness_evidence_overlay_issued": False,
    }


def _validate_readiness_state(
    value: object,
    name: str,
    _object_exact=_object,
    _nodes=READINESS_NODE_ORDER_V1,
    _all_false=_all_false_readiness_state_v2,
    _equal_exact=_exact_equal,
    _fail_exact=_fail,
) -> dict[str, object]:
    state = _object_exact(
        value,
        frozenset(
            {
                "node_states_in_canonical_topological_order",
                "satisfied_node_count",
                "highest_contiguous_satisfied_node",
                "all_nodes_satisfied",
                "readiness_evidence_overlay_issued",
            }
        ),
        name,
    )
    rows = state["node_states_in_canonical_topological_order"]
    if type(rows) is not list or len(rows) != len(_nodes):
        _fail_exact(FAIL_CLAIM_BOUNDARY_V2, f"{name} needs all sixteen readiness rows")
    for expected_node, raw in zip(_nodes, rows, strict=True):
        row = _object_exact(
            raw, frozenset({"node_id", "satisfied"}), f"{name} readiness row"
        )
        if row["node_id"] != expected_node or row["satisfied"] is not False:
            _fail_exact(FAIL_CLAIM_BOUNDARY_V2, f"{name} readiness row overclaims")
    if (
        type(state["satisfied_node_count"]) is not int
        or state["satisfied_node_count"] != 0
        or state["highest_contiguous_satisfied_node"] is not None
        or state["all_nodes_satisfied"] is not False
        or state["readiness_evidence_overlay_issued"] is not False
    ):
        _fail_exact(FAIL_CLAIM_BOUNDARY_V2, f"{name} readiness state must be all false")
    if not _equal_exact(state, _all_false()):
        _fail_exact(FAIL_CLAIM_BOUNDARY_V2, f"{name} readiness state differs")
    return state


def _validate_qv2_binding(
    value: object,
    name: str,
    _object_exact=_object,
    _equal_exact=_exact_equal,
    _expected=EXACT_QV2_BINDING_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    binding = _object_exact(
        value,
        frozenset(
            {
                "schema_version",
                "profile_id",
                "profile_payload_sha256",
                "source_sha256",
                "dependency_lock_sha256",
            }
        ),
        name,
    )
    if not _equal_exact(binding, _expected):
        _fail_exact(FAIL_BINDING_V2, f"{name} differs from exact frozen QV2 /9")
    return binding


def _validate_source_binding(
    value: object,
    _object_exact=_object,
    _text_exact=_text,
    _sha_exact=_sha,
    _commit_re=_COMMIT_RE,
    _fail_exact=_fail,
) -> dict[str, object]:
    binding = _object_exact(
        value,
        frozenset(
            {
                "source_commit_sha",
                "source_archive_sha256",
                "dependency_lock_sha256",
            }
        ),
        "source binding",
    )
    commit = _text_exact(binding["source_commit_sha"], "source commit SHA")
    if _commit_re.fullmatch(commit) is None or commit == "0" * 40:
        _fail_exact(FAIL_SCHEMA_V2, "source commit must be nonzero lowercase 40-hex")
    _sha_exact(binding["source_archive_sha256"], "source archive SHA")
    _sha_exact(binding["dependency_lock_sha256"], "dependency lock SHA")
    return binding


def _validate_role_policy(
    value: object,
    _object_exact=_object,
    _roles=ROLE_ORDER_V2,
    _services=SERVICE_ORDER_V2,
    _equal_exact=_exact_equal,
    _fail_exact=_fail,
) -> dict[str, object]:
    policy = _object_exact(
        value,
        frozenset(
            {
                "roles_in_exact_order",
                "role_count",
                "distinct_public_keys",
                "distinct_key_ids",
                "distinct_control_domains",
                "key_epoch",
                "purpose_policy_version",
                "services_in_exact_order",
                "service_binding_count",
                "distinct_service_uids_gids",
                "static_service_identity_only",
            }
        ),
        "role policy",
    )
    expected = {
        "roles_in_exact_order": list(_roles),
        "role_count": 8,
        "distinct_public_keys": True,
        "distinct_key_ids": True,
        "distinct_control_domains": True,
        "key_epoch": 1,
        "purpose_policy_version": 2,
        "services_in_exact_order": list(_services),
        "service_binding_count": 5,
        "distinct_service_uids_gids": True,
        "static_service_identity_only": True,
    }
    if not _equal_exact(policy, expected):
        _fail_exact(FAIL_SCHEMA_V2, "exact eight-role/five-service policy differs")
    return policy


def _validate_authority_profile_body(
    value: object,
    _object_exact=_object,
    _validate_owner=_validate_owner_binding,
    _validate_readiness=_validate_readiness_state,
    _validate_qv2=_validate_qv2_binding,
    _validate_source=_validate_source_binding,
    _validate_services=_validate_service_static_bindings,
    _validate_policy=_validate_role_policy,
    _validate_key=_validate_role_key,
    _services=SERVICE_ORDER_V2,
    _equal_exact=_exact_equal,
    _threat_model=_THREAT_MODEL_V2,
    _permanent_nonclaims=_PERMANENT_NONCLAIMS_V2,
    _fields=AUTHORITY_PROFILE_BODY_FIELDS_V2,
    _authority_profile_schema: str = AUTHORITY_PROFILE_SCHEMA_V2,
    _readiness_dag_id: str = READINESS_DAG_ID_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    body = _object_exact(
        value, _fields, "authority profile body"
    )
    if (
        body["schema_version"] != _authority_profile_schema
        or body["status"]
        != "CANDIDATE_STATIC_SERVICE_AUTHORITY_PROFILE_NOT_ADMITTED"
        or body["claim_level"]
        != (
            "STATIC_CANDIDATE_FOUNDATION_NOT_IMAGE_EXISTENCE_NOT_LIVE_"
            "RUNTIME_CLOSURE_NOT_AUTHORITY_NOT_READINESS"
        )
        or body["canonicalization"] != "RFC8785_EXACT_BYTES_NO_TRAILING_LF"
        or type(body["profile_epoch"]) is not int
        or body["profile_epoch"] != 2
    ):
        _fail_exact(FAIL_SCHEMA_V2, "authority profile V2 header differs")
    _validate_owner(body["owner_authorization_binding"])
    readiness_binding = _object_exact(
        body["readiness_dag_binding"],
        frozenset({"schema_version", "readiness_dag_id"}),
        "readiness DAG binding",
    )
    if readiness_binding != {
        "schema_version": (
            "hegel-machine-phase2b-actual-unsealed-960-readiness-dag/1"
        ),
        "readiness_dag_id": _readiness_dag_id,
    }:
        _fail_exact(FAIL_BINDING_V2, "readiness DAG binding differs")
    _validate_readiness(body["readiness_state"], "profile")
    _validate_qv2(body["qv2_binding"], "profile QV2 binding")
    _validate_source(body["source_binding"])
    services = _validate_services(
        body["service_static_bindings_in_exact_order"]
    )
    _validate_policy(body["role_policy"])
    governance_root = _validate_key(
        body["governance_root"], "GOVERNANCE_ANCHOR"
    )
    governance_service = services[_services.index("governance_status")]
    if (
        governance_service["role_key_id"] != governance_root["key_id"]
        or governance_service["key_epoch"] != governance_root["key_epoch"]
        or governance_service["authorized_signature_purposes"]
        != governance_root["authorized_signature_purposes"]
    ):
        _fail_exact(
            FAIL_BINDING_V2,
            "governance-status service does not bind the profile governance root",
        )
    if not _equal_exact(body["threat_model"], _threat_model):
        _fail_exact(FAIL_CLAIM_BOUNDARY_V2, "authority profile threat model differs")
    if not _equal_exact(body["permanent_nonclaims"], _permanent_nonclaims):
        _fail_exact(FAIL_CLAIM_BOUNDARY_V2, "authority profile nonclaims differ")
    return body


def _validate_stage_b_trust_lock_body(
    value: object,
    _object_exact=_object,
    _identifier_exact=_identifier,
    _sha_exact=_sha,
    _validate_qv2=_validate_qv2_binding,
    _fields=STAGE_B_TRUST_LOCK_BODY_FIELDS_V2,
    _lock_schema: str = STAGE_B_TRUST_LOCK_SCHEMA_V2,
    _authority_profile_schema: str = AUTHORITY_PROFILE_SCHEMA_V2,
    _registry_schema: str = SIGNER_REGISTRY_SCHEMA_V2,
    _authority_profile_prefix: str = AUTHORITY_PROFILE_PREFIX_V2,
    _registry_prefix: str = SIGNER_REGISTRY_PREFIX_V2,
    _fail_exact=_fail,
) -> dict[str, object]:
    body = _object_exact(
        value, _fields, "Stage-B trust-lock body"
    )
    if (
        body["schema_version"] != _lock_schema
        or body["status"] != "CANDIDATE_OUT_OF_BAND_STAGE_B_TRUST_LOCK"
        or body["claim_level"]
        != "CANDIDATE_GRAPH_PIN_ONLY_NOT_AUTHORITY_NOT_READINESS"
        or body["canonicalization"] != "RFC8785_EXACT_BYTES_NO_TRAILING_LF"
        or body["authority_profile_schema"] != _authority_profile_schema
        or body["signer_registry_schema"] != _registry_schema
    ):
        _fail_exact(FAIL_TRUST_ANCHOR_V2, "Stage-B trust-lock V2 header differs")
    _identifier_exact(
        body["authority_profile_id"],
        _authority_profile_prefix,
        "locked authority profile ID",
    )
    _sha_exact(
        body["authority_profile_payload_sha256"],
        "locked authority profile payload SHA",
    )
    _identifier_exact(
        body["signer_registry_id"],
        _registry_prefix,
        "locked signer registry ID",
    )
    _sha_exact(
        body["signer_registry_payload_sha256"],
        "locked signer registry payload SHA",
    )
    _validate_qv2(body["qv2_binding"], "Stage-B QV2 binding")
    return body


def _load_step5_stage_b_trust_lock_implementation_v2(
    payload: bytes,
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_payload_sha256_digest: bytes,
    _sha256=hashlib.sha256,
    _decode=decode_canonical_step5_json_v2,
    _object_exact=_object,
    _validate_body=_validate_stage_b_trust_lock_body,
    _content_id=phase2b_step5_content_id_v2,
    _identifier_exact=_identifier,
    _issue_artifact=_hidden_issue_verified_artifact_v2,
    _fail_exact=_fail,
) -> Phase2BStep5VerifiedArtifactV2:
    """Load a V2 lock using only raw independently provisioned byte pins.

    ``out_of_band_expected_lock_id_bytes`` is exact ASCII ID bytes and
    ``out_of_band_expected_payload_sha256_digest`` is the raw 32-byte digest,
    not hexadecimal text.  Copying either from ``payload`` does not establish
    an out-of-band trust root and violates this API's security contract.
    """

    if type(out_of_band_expected_lock_id_bytes) is not bytes:
        _fail_exact(FAIL_INPUT_V2, "out-of-band Stage-B lock ID must be exact bytes")
    if (
        type(out_of_band_expected_payload_sha256_digest) is not bytes
        or len(out_of_band_expected_payload_sha256_digest) != 32
        or out_of_band_expected_payload_sha256_digest == b"\x00" * 32
    ):
        _fail_exact(
            FAIL_INPUT_V2,
            "out-of-band Stage-B payload SHA must be one raw 32-byte digest",
        )
    try:
        expected_lock_id = out_of_band_expected_lock_id_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Phase2BStep5AuthorityErrorV2(
            FAIL_INPUT_V2, "out-of-band Stage-B lock ID is not exact ASCII"
        ) from exc
    _identifier_exact(
        expected_lock_id,
        STAGE_B_TRUST_LOCK_PREFIX_V2,
        "out-of-band Stage-B lock ID",
    )
    if (
        type(payload) is not bytes
        or not 1 <= len(payload) <= _MAX_CANONICAL_BYTES
        or payload.endswith(b"\n")
    ):
        _fail_exact(FAIL_INPUT_V2, "Stage-B payload must be bounded exact bytes")
    if _sha256(payload).digest() != out_of_band_expected_payload_sha256_digest:
        _fail_exact(
            FAIL_TRUST_ANCHOR_V2,
            "Stage-B payload differs from raw out-of-band SHA-256 pin",
        )
    decoded = _decode(payload)
    root = _object_exact(decoded, frozenset({"body", "id"}), "Stage-B trust lock")
    body = _validate_body(root["body"])
    expected_id = _content_id("stage_b_trust_lock", body)
    if (
        root["id"] != expected_id
        or expected_id != expected_lock_id
        or expected_id.encode("ascii") != out_of_band_expected_lock_id_bytes
    ):
        _fail_exact(
            FAIL_TRUST_ANCHOR_V2,
            "Stage-B trust lock differs from raw out-of-band ID pin",
        )
    return _issue_artifact("stage_b_trust_lock", expected_id, body)


def _verify_step5_authority_profile_decoded_v2(
    value: object,
    trust_lock: Phase2BStep5VerifiedArtifactV2,
    *,
    payload_sha256: str,
    _require_artifact=_hidden_require_verified_artifact_v2,
    _object_exact=_object,
    _validate_body=_validate_authority_profile_body,
    _content_id=phase2b_step5_content_id_v2,
    _equal_exact=_exact_equal,
    _issue_artifact=_hidden_issue_verified_artifact_v2,
    _fail_exact=_fail,
) -> Phase2BStep5VerifiedArtifactV2:
    lock = _require_artifact(
        trust_lock, "stage_b_trust_lock", "Stage-B trust lock"
    )
    root = _object_exact(value, frozenset({"body", "id"}), "authority profile")
    body = _validate_body(root["body"])
    expected_id = _content_id("authority_profile", body)
    if (
        root["id"] != expected_id
        or expected_id != lock.body["authority_profile_id"]
        or payload_sha256 != lock.body["authority_profile_payload_sha256"]
        or lock.body["authority_profile_schema"] != body["schema_version"]
        or not _equal_exact(body["qv2_binding"], lock.body["qv2_binding"])
    ):
        _fail_exact(
            FAIL_TRUST_ANCHOR_V2,
            "authority profile differs from the pinned Stage-B graph",
        )
    return _issue_artifact("authority_profile", expected_id, body)


def _load_step5_authority_profile_implementation_v2(
    payload: bytes,
    trust_lock: Phase2BStep5VerifiedArtifactV2,
    _decode=decode_canonical_step5_json_v2,
    _verify_decoded=_verify_step5_authority_profile_decoded_v2,
    _sha256=hashlib.sha256,
) -> Phase2BStep5VerifiedArtifactV2:
    """Load exact profile bytes under an already raw-pinned Stage-B lock."""

    return _verify_decoded(
        _decode(payload),
        trust_lock,
        payload_sha256=_sha256(payload).hexdigest(),
    )


def verify_step5_authority_profile_v2(*args: object, **kwargs: object) -> NoReturn:
    """Fail closed: decoded-object profile production verification is absent."""

    _fail(
        FAIL_TRUST_ANCHOR_V2,
        "use the bytes-only profile loader with a raw-pinned Stage-B lock",
    )


def _phase2b_step5_signature_preimage_implementation_v2(
    *,
    artifact_kind: str,
    body: object,
    role: str,
    purpose: str,
    signer_key_id: str,
    key_epoch: int = 1,
    _roles=ROLE_ORDER_V2,
    _purposes=ROLE_SIGNATURE_PURPOSES_V2,
    _domains=_SIGNATURE_DOMAINS,
    _identifier_exact=_identifier,
    _text_exact=_text,
    _canonical=canonical_step5_json_bytes_v2,
    _fail_exact=_fail,
) -> bytes:
    """Return the one domain-separated canonical V2 signature preimage."""

    if (
        type(role) is not str
        or role not in _roles
        or type(purpose) is not str
        or purpose not in _domains
        or purpose not in _purposes[role]
    ):
        _fail_exact(FAIL_INPUT_V2, "signature role/purpose pair is not authorized")
    _identifier_exact(signer_key_id, ROLE_KEY_PREFIX_V1, "signer role-key ID")
    if type(key_epoch) is not int or key_epoch != 1:
        _fail_exact(FAIL_INPUT_V2, "signature key epoch differs")
    subject = {
        "schema_version": "hegel-machine-phase2b-step5-signature-preimage/2",
        "artifact_kind": _text_exact(artifact_kind, "signature artifact kind"),
        "role": role,
        "purpose": purpose,
        "key_epoch": key_epoch,
        "signer_key_id": signer_key_id,
        "body": body,
    }
    return _domains[purpose] + _canonical(subject)


def _validate_detached_signature(
    value: object,
    role: str,
    purpose: str,
    key: Mapping[str, object],
    _object_exact=_object,
    _text_exact=_text,
    _fail_exact=_fail,
) -> dict[str, object]:
    signature = _object_exact(
        value, DETACHED_SIGNATURE_FIELDS_V2, "detached signature"
    )
    if (
        signature["schema_version"] != DETACHED_SIGNATURE_SCHEMA_V2
        or signature["algorithm"] != "Ed25519"
        or signature["role"] != role
        or signature["purpose"] != purpose
        or type(signature["key_epoch"]) is not int
        or signature["key_epoch"] != key["key_epoch"]
        or signature["signer_key_id"] != key["key_id"]
    ):
        _fail_exact(FAIL_SIGNATURE_V2, "detached signature metadata differs")
    signature_hex = _text_exact(signature["signature_hex"], "Ed25519 signature")
    if (
        len(signature_hex) != _SIGNATURE_HEX_LENGTH
        or re.fullmatch(r"[0-9a-f]+", signature_hex) is None
    ):
        _fail_exact(FAIL_SIGNATURE_V2, "Ed25519 signature encoding differs")
    return signature


def _verify_detached_signature(
    *,
    artifact_kind: str,
    body: object,
    signature: Mapping[str, object],
    key: Mapping[str, object],
    _backend_loader=(
        None
        if _Ed25519PublicKey is None
        else _Ed25519PublicKey.from_public_bytes
    ),
    _invalid_signature_type=_InvalidSignature,
    _preimage=_phase2b_step5_signature_preimage_implementation_v2,
    _fail_exact=_fail,
) -> str:
    if _backend_loader is None or _invalid_signature_type is None:
        _fail_exact(
            FAIL_SIGNATURE_BACKEND_V2,
            "cryptography Ed25519 verification backend is unavailable",
        )
    preimage = _preimage(
        artifact_kind=artifact_kind,
        body=body,
        role=signature["role"],  # type: ignore[arg-type]
        purpose=signature["purpose"],  # type: ignore[arg-type]
        signer_key_id=signature["signer_key_id"],  # type: ignore[arg-type]
        key_epoch=signature["key_epoch"],  # type: ignore[arg-type]
    )
    try:
        public_key = _backend_loader(
            bytes.fromhex(key["public_key_hex"])  # type: ignore[arg-type]
        )
        public_key.verify(
            bytes.fromhex(signature["signature_hex"]),  # type: ignore[arg-type]
            preimage,
        )
    except Exception as exc:
        raise Phase2BStep5AuthorityErrorV2(
            FAIL_SIGNATURE_V2, "Ed25519 signature rejected"
        ) from exc
    return "CRYPTOGRAPHY_ED25519"


def _registry_keys(
    registry: Phase2BStep5VerifiedArtifactV2,
    _require_artifact=_hidden_require_verified_artifact_v2,
    _fail_exact=_fail,
) -> dict[str, Mapping[str, object]]:
    verified = _require_artifact(registry, "signer_registry", "signer registry")
    rows = verified.body["role_keys_in_exact_order"]
    if type(rows) is not tuple or len(rows) != len(ROLE_ORDER_V2):
        _fail_exact(FAIL_TRUST_ANCHOR_V2, "verified registry token shape differs")
    return {row["role"]: row for row in rows}  # type: ignore[index]


def _validate_registry_service_key_links(
    profile: Phase2BStep5VerifiedArtifactV2,
    keys_by_role: Mapping[str, Mapping[str, object]],
    _services=SERVICE_ORDER_V2,
    _service_roles=SERVICE_ROLE_V2,
    _equal_exact=_exact_equal,
    _fail_exact=_fail,
) -> None:
    services = profile.body["service_static_bindings_in_exact_order"]
    if type(services) is not tuple or len(services) != len(_services):
        _fail_exact(FAIL_TRUST_ANCHOR_V2, "verified profile service shape differs")
    for expected_service, service in zip(_services, services, strict=True):
        role = _service_roles[expected_service]
        key = keys_by_role[role]
        if (
            service["service_name"] != expected_service
            or service["role"] != role
            or service["control_domain"] != key["control_domain"]
            or service["role_key_id"] != key["key_id"]
            or service["key_epoch"] != key["key_epoch"]
            or not _equal_exact(
                service["authorized_signature_purposes"],
                key["authorized_signature_purposes"],
            )
        ):
            _fail_exact(
                FAIL_BINDING_V2,
                f"{expected_service} service/registry role-key lineage differs",
            )


def _verify_step5_signer_registry_decoded_v2(
    value: object,
    trust_lock: Phase2BStep5VerifiedArtifactV2,
    profile: Phase2BStep5VerifiedArtifactV2,
    *,
    payload_sha256: str,
    _require_artifact=_hidden_require_verified_artifact_v2,
    _object_exact=_object,
    _validate_readiness=_validate_readiness_state,
    _validate_key=_validate_role_key,
    _equal_exact=_exact_equal,
    _validate_service_links=_validate_registry_service_key_links,
    _content_id=phase2b_step5_content_id_v2,
    _validate_signature=_validate_detached_signature,
    _verify_signature=_verify_detached_signature,
    _issue_artifact=_hidden_issue_verified_artifact_v2,
    _fail_exact=_fail,
) -> Phase2BStep5VerifiedArtifactV2:
    lock = _require_artifact(
        trust_lock, "stage_b_trust_lock", "Stage-B trust lock"
    )
    verified_profile = _require_artifact(
        profile, "authority_profile", "authority profile"
    )
    if verified_profile.artifact_id != lock.body["authority_profile_id"]:
        _fail_exact(
            FAIL_TRUST_ANCHOR_V2,
            "authority profile is outside the supplied Stage-B lock",
        )
    root = _object_exact(
        value,
        frozenset({"body", "id", "signatures"}),
        "signer registry",
    )
    body = _object_exact(
        root["body"], SIGNER_REGISTRY_BODY_FIELDS_V2, "signer registry body"
    )
    if (
        body["schema_version"] != SIGNER_REGISTRY_SCHEMA_V2
        or body["authority_profile_id"] != verified_profile.artifact_id
        or type(body["registry_epoch"]) is not int
        or body["registry_epoch"] != 2
        or body["status"] != "CANDIDATE_NOT_ADMITTED"
        or body["claim_level"]
        != "SIGNED_STATIC_KEY_REGISTRY_CANDIDATE_NOT_AUTHORITY_NOT_READINESS"
        or type(body["revocation_epoch"]) is not int
        or body["revocation_epoch"] != 0
    ):
        _fail_exact(FAIL_SCHEMA_V2, "signer registry V2 header differs")
    _validate_readiness(body["readiness_state"], "signer registry")
    raw_keys = body["role_keys_in_exact_order"]
    if type(raw_keys) is not list or len(raw_keys) != len(ROLE_ORDER_V2):
        _fail_exact(FAIL_SCHEMA_V2, "signer registry needs exactly eight role keys")
    keys = [
        _validate_key(row, role)
        for role, row in zip(ROLE_ORDER_V2, raw_keys, strict=True)
    ]
    if not _equal_exact(keys[0], verified_profile.body["governance_root"]):
        _fail_exact(
            FAIL_BINDING_V2,
            "registry governance key differs from the profile governance root",
        )
    for field in ("public_key_hex", "key_id", "control_domain"):
        if len({key[field] for key in keys}) != len(ROLE_ORDER_V2):
            _fail_exact(
                FAIL_SCHEMA_V2,
                f"registry {field} values are not pairwise distinct",
            )
    keys_by_role: dict[str, Mapping[str, object]] = {
        key["role"]: key for key in keys  # type: ignore[misc]
    }
    _validate_service_links(verified_profile, keys_by_role)
    expected_id = _content_id("signer_registry", body)
    if (
        root["id"] != expected_id
        or expected_id != lock.body["signer_registry_id"]
        or payload_sha256 != lock.body["signer_registry_payload_sha256"]
        or lock.body["signer_registry_schema"] != body["schema_version"]
    ):
        _fail_exact(
            FAIL_TRUST_ANCHOR_V2,
            "signer registry differs from the pinned Stage-B graph",
        )
    signatures = root["signatures"]
    if type(signatures) is not list or len(signatures) != 1:
        _fail_exact(
            FAIL_SIGNATURE_V2,
            "signer registry needs exactly one governance signature",
        )
    signature = _validate_signature(
        signatures[0],
        "GOVERNANCE_ANCHOR",
        "SIGN_SIGNER_REGISTRY",
        keys[0],
    )
    backend = _verify_signature(
        artifact_kind="signer_registry",
        body=body,
        signature=signature,
        key=keys[0],
    )
    return _issue_artifact("signer_registry", expected_id, body, (backend,))


def _load_step5_signer_registry_implementation_v2(
    payload: bytes,
    trust_lock: Phase2BStep5VerifiedArtifactV2,
    profile: Phase2BStep5VerifiedArtifactV2,
    _decode=decode_canonical_step5_json_v2,
    _verify_decoded=_verify_step5_signer_registry_decoded_v2,
    _sha256=hashlib.sha256,
) -> Phase2BStep5VerifiedArtifactV2:
    """Load exact governance-signed registry bytes under the closed lock."""

    return _verify_decoded(
        _decode(payload),
        trust_lock,
        profile,
        payload_sha256=_sha256(payload).hexdigest(),
    )


def verify_step5_signer_registry_v2(*args: object, **kwargs: object) -> NoReturn:
    """Fail closed: decoded objects and injected verifiers are unsupported."""

    _fail(
        FAIL_TRUST_ANCHOR_V2,
        "use the bytes-only registry loader with the raw-pinned Stage-B graph",
    )


def _require_locked_step5_authority_graph_implementation_v2(
    trust_lock: Phase2BStep5VerifiedArtifactV2,
    profile: Phase2BStep5VerifiedArtifactV2,
    registry: Phase2BStep5VerifiedArtifactV2,
    _require_artifact=_hidden_require_verified_artifact_v2,
    _equal_exact=_exact_equal,
    _all_false=_all_false_readiness_state_v2,
    _validate_service_links=_validate_registry_service_key_links,
    _registry_keys_exact=_registry_keys,
    _fail_exact=_fail,
) -> tuple[
    Phase2BStep5VerifiedArtifactV2,
    Phase2BStep5VerifiedArtifactV2,
    Phase2BStep5VerifiedArtifactV2,
]:
    """Re-establish the complete lock/profile/registry/service-key lineage."""

    lock = _require_artifact(
        trust_lock, "stage_b_trust_lock", "Stage-B trust lock"
    )
    verified_profile = _require_artifact(
        profile, "authority_profile", "authority profile"
    )
    verified_registry = _require_artifact(
        registry, "signer_registry", "signer registry"
    )
    if (
        verified_profile.artifact_id != lock.body["authority_profile_id"]
        or verified_registry.artifact_id != lock.body["signer_registry_id"]
        or verified_registry.body["authority_profile_id"]
        != verified_profile.artifact_id
        or lock.body["authority_profile_schema"]
        != verified_profile.body["schema_version"]
        or lock.body["signer_registry_schema"]
        != verified_registry.body["schema_version"]
        or not _equal_exact(
            verified_profile.body["qv2_binding"], lock.body["qv2_binding"]
        )
    ):
        _fail_exact(
            FAIL_TRUST_ANCHOR_V2,
            "Stage-B lock/profile/registry lineage is not closed",
        )
    if (
        not _equal_exact(
            verified_profile.body["readiness_state"],
            _all_false(),
        )
        or not _equal_exact(
            verified_registry.body["readiness_state"],
            _all_false(),
        )
    ):
        _fail_exact(
            FAIL_CLAIM_BOUNDARY_V2,
            "closed Stage-B graph readiness state is not exactly all false",
        )
    _validate_service_links(
        verified_profile, _registry_keys_exact(verified_registry)
    )
    return lock, verified_profile, verified_registry


READINESS_CANDIDATE_SNAPSHOT_BODY_FIELDS_V2: Final = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "stage_b_trust_lock_id",
        "authority_profile_id",
        "signer_registry_id",
        "readiness_dag_id",
        "service_binding_ids_in_exact_order",
        "node_states_in_canonical_topological_order",
        "satisfied_node_count",
        "highest_contiguous_satisfied_node",
        "all_nodes_satisfied",
        "actual_campaign_id",
        "actual_attempt_id",
        "actual_input_consumed",
        "service_images_exist",
        "service_images_executed",
        "static_service_bindings_live_verified",
        "live_runtime_closure_verified",
        "live_distinct_service_uids_gids_verified",
        "cas_store_verified",
        "logical_time_verified",
        "governance_status_ledger_admitted",
        "latest_owner_authorization_status_verified",
        "owner_status_external_anti_fork_anchor_verified",
        "r01_authorization_status_receipt_issued",
        "r01_satisfied",
        "r02_or_later_satisfied",
        "readiness_evidence_overlay_issued",
        "execution_authorized",
    }
)


def _validate_readiness_dag(value: object) -> None:
    wrapper = _object(
        value,
        frozenset({"readiness_dag_body", "readiness_dag_id"}),
        "readiness DAG",
    )
    if wrapper["readiness_dag_id"] != READINESS_DAG_ID_V1:
        _fail(FAIL_BINDING_V2, "readiness DAG ID differs")
    body = _object(
        wrapper["readiness_dag_body"],
        frozenset(
            {
                "schema_version",
                "document_id",
                "status",
                "claim_level",
                "canonicalization",
                "exact_contract_bindings",
                "ordering_erratum",
                "readiness_dag",
                "safety_boundary",
                "wire_covert_qualification_contract",
            }
        ),
        "readiness DAG body",
    )
    expected_id = "phase2b_actual_unsealed_960_readiness_dag_v1_" + hashlib.sha256(
        b"HEGEL/PHASE2B/ACTUAL_UNSEALED_960/READINESS_DAG/BODY/V1\x00"
        + canonical_step5_json_bytes_v2(body)
    ).hexdigest()
    if expected_id != READINESS_DAG_ID_V1:
        _fail(FAIL_CONTENT_ID_V2, "readiness DAG body content address differs")
    dag = _object(
        body["readiness_dag"],
        frozenset(
            {
                "all_nodes_satisfied",
                "edge_semantics",
                "node_count",
                "nodes_in_canonical_topological_order",
                "satisfied_node_count",
            }
        ),
        "readiness DAG graph",
    )
    nodes = dag["nodes_in_canonical_topological_order"]
    if (
        dag["all_nodes_satisfied"] is not False
        or type(dag["satisfied_node_count"]) is not int
        or dag["satisfied_node_count"] != 0
        or type(dag["node_count"]) is not int
        or dag["node_count"] != len(READINESS_NODE_ORDER_V1)
        or type(nodes) is not list
        or len(nodes) != len(READINESS_NODE_ORDER_V1)
    ):
        _fail(
            FAIL_CLAIM_BOUNDARY_V2,
            "base readiness DAG must contain sixteen unsatisfied nodes",
        )
    for index, raw in enumerate(nodes):
        node = _object(
            raw,
            frozenset(
                {
                    "node_id",
                    "dependencies",
                    "evidence_requirement",
                    "initial_satisfied",
                }
            ),
            "readiness DAG node",
        )
        if (
            node["node_id"] != READINESS_NODE_ORDER_V1[index]
            or node["dependencies"] != list(READINESS_DEPENDENCIES_V1[index])
            or node["initial_satisfied"] is not False
        ):
            _fail(FAIL_BINDING_V2, "readiness DAG topology differs")


def _derive_step5_readiness_candidate_snapshot_implementation_v2(
    trust_lock: Phase2BStep5VerifiedArtifactV2,
    profile: Phase2BStep5VerifiedArtifactV2,
    registry: Phase2BStep5VerifiedArtifactV2,
    readiness_dag_payload: bytes,
    _require_graph=_require_locked_step5_authority_graph_implementation_v2,
    _decode=decode_canonical_step5_json_v2,
    _validate_dag=_validate_readiness_dag,
    _object_exact=_object,
    _content_id=phase2b_step5_content_id_v2,
    _issue_artifact=_hidden_issue_verified_artifact_v2,
) -> Phase2BStep5VerifiedArtifactV2:
    """Derive a non-evidence snapshot in which every readiness row is false."""

    lock, verified_profile, verified_registry = (
        _require_graph(trust_lock, profile, registry)
    )
    _validate_dag(
        _decode(readiness_dag_payload)
    )
    services = verified_profile.body[
        "service_static_bindings_in_exact_order"
    ]
    assert type(services) is tuple
    body = {
        "schema_version": READINESS_CANDIDATE_SNAPSHOT_SCHEMA_V2,
        "status": "CANDIDATE_FOUNDATION_ONLY",
        "claim_level": "ALL_SIXTEEN_READINESS_NODES_FALSE_NOT_EVIDENCE",
        "stage_b_trust_lock_id": lock.artifact_id,
        "authority_profile_id": verified_profile.artifact_id,
        "signer_registry_id": verified_registry.artifact_id,
        "readiness_dag_id": READINESS_DAG_ID_V1,
        "service_binding_ids_in_exact_order": [
            service["binding_id"] for service in services
        ],
        "node_states_in_canonical_topological_order": [
            {"node_id": node, "satisfied": False}
            for node in READINESS_NODE_ORDER_V1
        ],
        "satisfied_node_count": 0,
        "highest_contiguous_satisfied_node": None,
        "all_nodes_satisfied": False,
        "actual_campaign_id": None,
        "actual_attempt_id": None,
        "actual_input_consumed": False,
        "service_images_exist": False,
        "service_images_executed": False,
        "static_service_bindings_live_verified": False,
        "live_runtime_closure_verified": False,
        "live_distinct_service_uids_gids_verified": False,
        "cas_store_verified": False,
        "logical_time_verified": False,
        "governance_status_ledger_admitted": False,
        "latest_owner_authorization_status_verified": False,
        "owner_status_external_anti_fork_anchor_verified": False,
        "r01_authorization_status_receipt_issued": False,
        "r01_satisfied": False,
        "r02_or_later_satisfied": False,
        "readiness_evidence_overlay_issued": False,
        "execution_authorized": False,
    }
    _object_exact(
        body,
        READINESS_CANDIDATE_SNAPSHOT_BODY_FIELDS_V2,
        "readiness candidate snapshot body",
    )
    snapshot_id = _content_id("readiness_candidate", body)
    return _issue_artifact("readiness_candidate", snapshot_id, body)


def _definition_captured_function_graph_v2() -> dict[str, object]:
    """Copy module functions onto a private definition-time global namespace.

    Python function objects normally resolve otherwise-unqualified names through
    their module dictionary.  Merely retaining the root loader function is
    therefore insufficient: replacing an exported schema constant or a helper
    attribute could otherwise redefine a deep validation edge.  This routine
    creates one private namespace, clones every locally defined function onto
    it, rewrites function-valued defaults to the corresponding private clone,
    and returns that closed graph.  Imported RFC-8785 and Ed25519 callables and
    all wire constants are snapshots in the same namespace.

    The boundary is deliberate ordinary-Python module rebinding resistance.  It
    does not claim resistance to function ``__globals__`` introspection or
    arbitrary interpreter-memory mutation.
    """

    module_namespace = globals()
    private_namespace: dict[str, object] = dict(module_namespace)
    local_functions = {
        name: value
        for name, value in module_namespace.items()
        if type(value) is FunctionType and value.__module__ == __name__
    }

    provisional: dict[str, FunctionType] = {}
    by_identity: dict[int, FunctionType] = {}
    for name, implementation in local_functions.items():
        clone = FunctionType(
            implementation.__code__,
            private_namespace,
            implementation.__name__,
            implementation.__defaults__,
            implementation.__closure__,
        )
        provisional[name] = clone
        by_identity[id(implementation)] = clone
    private_namespace.update(provisional)

    def captured_default(value: object) -> object:
        return by_identity.get(id(value), value)

    captured: dict[str, FunctionType] = {}
    for name, implementation in local_functions.items():
        defaults = implementation.__defaults__
        captured_defaults = (
            None
            if defaults is None
            else tuple(captured_default(value) for value in defaults)
        )
        clone = FunctionType(
            implementation.__code__,
            private_namespace,
            implementation.__name__,
            captured_defaults,
            implementation.__closure__,
        )
        clone.__kwdefaults__ = (
            None
            if implementation.__kwdefaults__ is None
            else {
                key: captured_default(value)
                for key, value in implementation.__kwdefaults__.items()
            }
        )
        clone.__annotations__ = dict(implementation.__annotations__)
        clone.__doc__ = implementation.__doc__
        clone.__module__ = implementation.__module__
        clone.__qualname__ = implementation.__qualname__
        captured[name] = clone

    final_by_identity = {
        id(local_functions[name]): clone for name, clone in captured.items()
    }

    def final_captured_default(value: object) -> object:
        return final_by_identity.get(id(value), value)

    for name, implementation in local_functions.items():
        clone = captured[name]
        defaults = implementation.__defaults__
        clone.__defaults__ = (
            None
            if defaults is None
            else tuple(final_captured_default(value) for value in defaults)
        )
        clone.__kwdefaults__ = (
            None
            if implementation.__kwdefaults__ is None
            else {
                key: final_captured_default(value)
                for key, value in implementation.__kwdefaults__.items()
            }
        )

    private_namespace.update(captured)
    return private_namespace


def _build_public_authority_entrypoints_v2():
    """Capture the bytes-only call graph and discard module-global issuers."""

    captured = _definition_captured_function_graph_v2()
    load_lock_implementation = captured[
        "_load_step5_stage_b_trust_lock_implementation_v2"
    ]
    load_profile_implementation = captured[
        "_load_step5_authority_profile_implementation_v2"
    ]
    load_registry_implementation = captured[
        "_load_step5_signer_registry_implementation_v2"
    ]
    require_graph_implementation = captured[
        "_require_locked_step5_authority_graph_implementation_v2"
    ]
    derive_snapshot_implementation = captured[
        "_derive_step5_readiness_candidate_snapshot_implementation_v2"
    ]
    signature_preimage_implementation = captured[
        "_phase2b_step5_signature_preimage_implementation_v2"
    ]
    content_id_implementation = captured["phase2b_step5_content_id_v2"]
    role_key_id_implementation = captured[
        "phase2b_step5_role_key_id_v2"
    ]
    service_binding_id_implementation = captured[
        "phase2b_step5_service_static_binding_id_v2"
    ]
    canonical_implementation = captured["canonical_step5_json_bytes_v2"]
    decode_implementation = captured["decode_canonical_step5_json_v2"]

    def public_canonical_step5_json_bytes_v2(value: object) -> bytes:
        return canonical_implementation(value)

    def public_decode_canonical_step5_json_v2(
        payload: bytes, *, maximum_bytes: int = _MAX_CANONICAL_BYTES
    ) -> object:
        return decode_implementation(payload, maximum_bytes=maximum_bytes)

    def public_phase2b_step5_content_id_v2(kind: str, body: object) -> str:
        return content_id_implementation(kind, body)

    def public_phase2b_step5_role_key_id_v2(
        row_without_or_with_id: object,
    ) -> str:
        return role_key_id_implementation(row_without_or_with_id)

    def public_phase2b_step5_service_static_binding_id_v2(
        binding_without_or_with_id: object,
    ) -> str:
        return service_binding_id_implementation(binding_without_or_with_id)

    def load_step5_stage_b_trust_lock_v2(
        payload: bytes,
        *,
        out_of_band_expected_lock_id_bytes: bytes,
        out_of_band_expected_payload_sha256_digest: bytes,
    ) -> Phase2BStep5VerifiedArtifactV2:
        return load_lock_implementation(
            payload,
            out_of_band_expected_lock_id_bytes=(
                out_of_band_expected_lock_id_bytes
            ),
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_payload_sha256_digest
            ),
        )

    def load_step5_authority_profile_v2(
        payload: bytes,
        trust_lock: Phase2BStep5VerifiedArtifactV2,
    ) -> Phase2BStep5VerifiedArtifactV2:
        return load_profile_implementation(payload, trust_lock)

    def load_step5_signer_registry_v2(
        payload: bytes,
        trust_lock: Phase2BStep5VerifiedArtifactV2,
        profile: Phase2BStep5VerifiedArtifactV2,
    ) -> Phase2BStep5VerifiedArtifactV2:
        return load_registry_implementation(payload, trust_lock, profile)

    def require_locked_step5_authority_graph_v2(
        trust_lock: Phase2BStep5VerifiedArtifactV2,
        profile: Phase2BStep5VerifiedArtifactV2,
        registry: Phase2BStep5VerifiedArtifactV2,
    ) -> tuple[
        Phase2BStep5VerifiedArtifactV2,
        Phase2BStep5VerifiedArtifactV2,
        Phase2BStep5VerifiedArtifactV2,
    ]:
        return require_graph_implementation(trust_lock, profile, registry)

    def derive_step5_readiness_candidate_snapshot_v2(
        trust_lock: Phase2BStep5VerifiedArtifactV2,
        profile: Phase2BStep5VerifiedArtifactV2,
        registry: Phase2BStep5VerifiedArtifactV2,
        readiness_dag_payload: bytes,
    ) -> Phase2BStep5VerifiedArtifactV2:
        return derive_snapshot_implementation(
            trust_lock,
            profile,
            registry,
            readiness_dag_payload,
        )

    def phase2b_step5_signature_preimage_v2(
        *,
        artifact_kind: str,
        body: object,
        role: str,
        purpose: str,
        signer_key_id: str,
        key_epoch: int = 1,
    ) -> bytes:
        return signature_preimage_implementation(
            artifact_kind=artifact_kind,
            body=body,
            role=role,
            purpose=purpose,
            signer_key_id=signer_key_id,
            key_epoch=key_epoch,
        )

    return (
        load_step5_stage_b_trust_lock_v2,
        load_step5_authority_profile_v2,
        load_step5_signer_registry_v2,
        require_locked_step5_authority_graph_v2,
        derive_step5_readiness_candidate_snapshot_v2,
        phase2b_step5_signature_preimage_v2,
        public_phase2b_step5_content_id_v2,
        public_phase2b_step5_role_key_id_v2,
        public_phase2b_step5_service_static_binding_id_v2,
        public_canonical_step5_json_bytes_v2,
        public_decode_canonical_step5_json_v2,
    )


(
    load_step5_stage_b_trust_lock_v2,
    load_step5_authority_profile_v2,
    load_step5_signer_registry_v2,
    require_locked_step5_authority_graph_v2,
    derive_step5_readiness_candidate_snapshot_v2,
    phase2b_step5_signature_preimage_v2,
    phase2b_step5_content_id_v2,
    phase2b_step5_role_key_id_v2,
    phase2b_step5_service_static_binding_id_v2,
    canonical_step5_json_bytes_v2,
    decode_canonical_step5_json_v2,
) = _build_public_authority_entrypoints_v2()
del _build_public_authority_entrypoints_v2
del _definition_captured_function_graph_v2
del _load_step5_stage_b_trust_lock_implementation_v2
del _load_step5_authority_profile_implementation_v2
del _load_step5_signer_registry_implementation_v2
del _verify_step5_authority_profile_decoded_v2
del _verify_step5_signer_registry_decoded_v2
del _require_locked_step5_authority_graph_implementation_v2
del _derive_step5_readiness_candidate_snapshot_implementation_v2
del _phase2b_step5_signature_preimage_implementation_v2
del _hidden_issue_verified_artifact_v2
del _hidden_require_verified_artifact_v2


def verify_step5_runtime_closure_v2(*args: object, **kwargs: object) -> NoReturn:
    """Fail closed: static policy hashes are not observed runtime closure."""

    _fail(
        FAIL_NOT_IMPLEMENTED_V2,
        "live service runtime-closure verification is not implemented",
    )


def verify_step5_r01_authorization_status_receipt_v2(
    *args: object, **kwargs: object
) -> NoReturn:
    """Fail closed: the future source/pure R01 receipt does not yet exist."""

    _fail(
        FAIL_NOT_IMPLEMENTED_V2,
        "real R01 authorization-status receipt verification is not implemented",
    )


def derive_step5_r01_readiness_overlay_v2(
    *args: object, **kwargs: object
) -> NoReturn:
    """Fail closed: no candidate artifact may set R01 or readiness true."""

    _fail(
        FAIL_NOT_IMPLEMENTED_V2,
        "R01/readiness overlay issuance is disabled; every readiness row remains false",
    )


__all__ = [
    "ACTOR_PROFILE_CANONICAL_SHA256_V1",
    "ACTOR_PROFILE_CONTENT_ID_V1",
    "AUTHORITY_CLASS_V1",
    "AUTHORITY_PROFILE_BODY_FIELDS_V2",
    "AUTHORITY_PROFILE_PREFIX_V2",
    "AUTHORITY_PROFILE_SCHEMA_V2",
    "AUTHORITY_REQUIREMENT_ID_V1",
    "DETACHED_SIGNATURE_FIELDS_V2",
    "DETACHED_SIGNATURE_SCHEMA_V2",
    "EXACT_FREEZE_ID_V1",
    "EXACT_FREEZE_VERSION_V1",
    "EXACT_QV2_BINDING_V1",
    "FAIL_BINDING_V2",
    "FAIL_CANONICAL_V2",
    "FAIL_CLAIM_BOUNDARY_V2",
    "FAIL_CONTENT_ID_V2",
    "FAIL_INPUT_V2",
    "FAIL_NOT_IMPLEMENTED_V2",
    "FAIL_SCHEMA_V2",
    "FAIL_SIGNATURE_BACKEND_V2",
    "FAIL_SIGNATURE_V2",
    "FAIL_TRUST_ANCHOR_V2",
    "LOGICAL_STAGE_WIRE_POLICY_ID_V1",
    "LOGICAL_STAGE_WIRE_SCHEMA_ID_V1",
    "OWNER_AUTHORIZATION_CONFIG_SHA256_V1",
    "OWNER_AUTHORIZATION_ID_V1",
    "OWNER_AUTHORIZATION_SOURCE_SHA256_V1",
    "PROTOCOL_ID_V1",
    "Phase2BStep5AuthorityErrorV2",
    "Phase2BStep5VerifiedArtifactV2",
    "QV2_DEPENDENCY_LOCK_SHA256_V1",
    "QV2_PROFILE_ID_V1",
    "QV2_PROFILE_PAYLOAD_SHA256_V1",
    "QV2_SOURCE_SHA256_V1",
    "READINESS_CANDIDATE_PREFIX_V2",
    "READINESS_CANDIDATE_SNAPSHOT_BODY_FIELDS_V2",
    "READINESS_CANDIDATE_SNAPSHOT_SCHEMA_V2",
    "READINESS_DAG_ID_V1",
    "READINESS_NODE_ORDER_V1",
    "ROLE_CONTROL_DOMAINS_V2",
    "ROLE_KEY_FIELDS_V2",
    "ROLE_KEY_PREFIX_V1",
    "ROLE_KEY_SCHEMA_V1",
    "ROLE_ORDER_V2",
    "ROLE_SIGNATURE_PURPOSES_V2",
    "SERVICE_ORDER_V2",
    "SERVICE_ROLE_V2",
    "SERVICE_STATIC_BINDING_FIELDS_V2",
    "SERVICE_STATIC_BINDING_PREFIX_V2",
    "SERVICE_STATIC_BINDING_SCHEMA_V2",
    "SERVICE_TYPES_V2",
    "SERVICE_UID_GID_V2",
    "SIGNER_REGISTRY_BODY_FIELDS_V2",
    "SIGNER_REGISTRY_PREFIX_V2",
    "SIGNER_REGISTRY_SCHEMA_V2",
    "STAGE_B_TRUST_LOCK_BODY_FIELDS_V2",
    "STAGE_B_TRUST_LOCK_PREFIX_V2",
    "STAGE_B_TRUST_LOCK_SCHEMA_V2",
    "canonical_step5_json_bytes_v2",
    "decode_canonical_step5_json_v2",
    "derive_step5_r01_readiness_overlay_v2",
    "derive_step5_readiness_candidate_snapshot_v2",
    "load_step5_authority_profile_v2",
    "load_step5_signer_registry_v2",
    "load_step5_stage_b_trust_lock_v2",
    "phase2b_step5_content_id_v2",
    "phase2b_step5_role_key_id_v2",
    "phase2b_step5_service_static_binding_id_v2",
    "phase2b_step5_signature_preimage_v2",
    "require_locked_step5_authority_graph_v2",
    "verify_step5_authority_profile_v2",
    "verify_step5_r01_authorization_status_receipt_v2",
    "verify_step5_runtime_closure_v2",
    "verify_step5_signer_registry_v2",
]
