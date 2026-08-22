"""Pure Step-5 authority foundation for Phase-2B wire/covert qualification.

The module verifies caller-supplied RFC-8785 objects only.  It performs no
filesystem, container, network, clock, key-generation, signing, qualification,
or campaign operation.  Its authority model is deliberately limited to eight
owner-accepted technical roles controlled by the same administrator; successful
verification is not evidence of organizational or human independence.

V1 covers a fail-closed bootstrap graph only: an out-of-band-pinned Stage-B
trust lock, authority profile, governance-signed role registry, and dual-signed
R01/R02 *candidates*.  Candidates never satisfy readiness nodes.  A real R01
requires a governance status ledger plus durable CAS and logical time; those
services, real R01/R02 receipts, readiness evidence overlays, and R03--R05
receipt semantics are intentionally outside this foundation revision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from types import MappingProxyType
from typing import Final, Mapping, NoReturn

from .rfc8785_v1 import (
    RFC8785Error,
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)

try:  # Optional backend.  Default verification fails closed when unavailable.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - environment dependent.
    _Ed25519PublicKey = None


AUTHORITY_PROFILE_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-authority-profile/1"
)
STAGE_B_TRUST_LOCK_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-stage-b-trust-lock/1"
)
SIGNER_REGISTRY_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-signer-registry/1"
)
DETACHED_SIGNATURE_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-detached-signature/1"
)
R01_AUTHORIZATION_STATUS_CANDIDATE_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-r01-authorization-status-candidate/1"
)
R02_SYNTHETIC_BRANCH_CANDIDATE_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-r02-synthetic-branch-candidate/1"
)
READINESS_CANDIDATE_SNAPSHOT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-readiness-candidate-snapshot/1"
)

AUTHORITY_PROFILE_PREFIX_V1: Final = "phase2b_step5_authority_profile_v1_"
STAGE_B_TRUST_LOCK_PREFIX_V1: Final = "phase2b_step5_stage_b_trust_lock_v1_"
SIGNER_REGISTRY_PREFIX_V1: Final = "phase2b_step5_signer_registry_v1_"
ROLE_KEY_PREFIX_V1: Final = "phase2b_step5_role_key_v1_"
R01_CANDIDATE_PREFIX_V1: Final = "phase2b_step5_r01_authorization_status_candidate_v1_"
R02_CANDIDATE_PREFIX_V1: Final = "phase2b_step5_r02_synthetic_branch_candidate_v1_"
READINESS_CANDIDATE_PREFIX_V1: Final = "phase2b_step5_readiness_candidate_v1_"

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
_EXACT_QV2_BINDING_V1: Final = {
    "schema_version": "hegel-machine-phase2b-formal-wire-covert-qualification/9",
    "profile_id": QV2_PROFILE_ID_V1,
    "profile_payload_sha256": QV2_PROFILE_PAYLOAD_SHA256_V1,
    "source_sha256": QV2_SOURCE_SHA256_V1,
    "dependency_lock_sha256": QV2_DEPENDENCY_LOCK_SHA256_V1,
}

ROLE_ORDER_V1: Final = (
    "GOVERNANCE_ANCHOR",
    "SOURCE_RUNTIME_CUSTODIAN",
    "WIRE_QUALIFIER_R03",
    "AUDITOR_QUALIFIER_R04",
    "INTEGRATION_QUALIFIER_R05",
    "CAS_STORE",
    "LOGICAL_TIME",
    "PURE_REPLAYER",
)

ROLE_CONTROL_DOMAINS_V1: Final = MappingProxyType(
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

ROLE_SIGNATURE_PURPOSES_V1: Final = MappingProxyType(
    {
        "GOVERNANCE_ANCHOR": ("SIGN_SIGNER_REGISTRY",),
        "SOURCE_RUNTIME_CUSTODIAN": (
            "SIGN_R01_AUTHORIZATION_STATUS_CANDIDATE",
            "SIGN_R02_SYNTHETIC_BRANCH_CANDIDATE",
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
        ),
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

FAIL_INPUT_V1: Final = "FAIL_STEP5_AUTHORITY_V1_INPUT"
FAIL_CANONICAL_V1: Final = "FAIL_STEP5_AUTHORITY_V1_CANONICAL"
FAIL_SCHEMA_V1: Final = "FAIL_STEP5_AUTHORITY_V1_SCHEMA"
FAIL_CONTENT_ID_V1: Final = "FAIL_STEP5_AUTHORITY_V1_CONTENT_ID"
FAIL_BINDING_V1: Final = "FAIL_STEP5_AUTHORITY_V1_BINDING"
FAIL_SIGNATURE_BACKEND_V1: Final = "FAIL_STEP5_AUTHORITY_V1_SIGNATURE_BACKEND"
FAIL_SIGNATURE_V1: Final = "FAIL_STEP5_AUTHORITY_V1_SIGNATURE"
FAIL_CLAIM_BOUNDARY_V1: Final = "FAIL_STEP5_AUTHORITY_V1_CLAIM_BOUNDARY"
FAIL_TRUST_ANCHOR_V1: Final = "FAIL_STEP5_AUTHORITY_V1_TRUST_ANCHOR"
FAIL_NOT_IMPLEMENTED_V1: Final = "FAIL_STEP5_AUTHORITY_V1_NOT_IMPLEMENTED"

_MAX_CANONICAL_BYTES: Final = 4 * 1024 * 1024
_MAX_DEPTH: Final = 64
_MAX_NODES: Final = 200_000
_MAX_STRING_UTF8_BYTES: Final = 1 * 1024 * 1024
_SAFE_INTEGER: Final = (1 << 53) - 1
_PUBLIC_KEY_HEX_LENGTH: Final = 64
_SIGNATURE_HEX_LENGTH: Final = 128
_HEX64_RE: Final = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}")
_IMAGE_RE: Final = re.compile(r"sha256:[0-9a-f]{64}")
_ID_RE: Final = re.compile(r"[a-z0-9][a-z0-9_./:-]{0,255}_[0-9a-f]{64}")

_DOMAINS: Final = MappingProxyType(
    {
        "stage_b_trust_lock": b"HEGEL/PHASE2B/STEP5/STAGE_B_TRUST_LOCK/V1\x00",
        "authority_profile": b"HEGEL/PHASE2B/STEP5/AUTHORITY_PROFILE/V1\x00",
        "signer_registry": b"HEGEL/PHASE2B/STEP5/SIGNER_REGISTRY/V1\x00",
        "role_key": b"HEGEL/PHASE2B/STEP5/ROLE_KEY/V1\x00",
        "r01_candidate": b"HEGEL/PHASE2B/STEP5/R01_STATUS_CANDIDATE/V1\x00",
        "r02_candidate": b"HEGEL/PHASE2B/STEP5/R02_BRANCH_CANDIDATE/V1\x00",
        "readiness_candidate": b"HEGEL/PHASE2B/STEP5/READINESS_CANDIDATE/V1\x00",
    }
)
_PREFIXES: Final = MappingProxyType(
    {
        "stage_b_trust_lock": STAGE_B_TRUST_LOCK_PREFIX_V1,
        "authority_profile": AUTHORITY_PROFILE_PREFIX_V1,
        "signer_registry": SIGNER_REGISTRY_PREFIX_V1,
        "role_key": ROLE_KEY_PREFIX_V1,
        "r01_candidate": R01_CANDIDATE_PREFIX_V1,
        "r02_candidate": R02_CANDIDATE_PREFIX_V1,
        "readiness_candidate": READINESS_CANDIDATE_PREFIX_V1,
    }
)
_SIGNATURE_DOMAINS: Final = MappingProxyType(
    {
        "SIGN_SIGNER_REGISTRY": b"HEGEL/PHASE2B/STEP5/SIGN/REGISTRY/V1\x00",
        "SIGN_R01_AUTHORIZATION_STATUS_CANDIDATE": b"HEGEL/PHASE2B/STEP5/SIGN/R01_CANDIDATE/CUSTODIAN/V1\x00",
        "COUNTERSIGN_R01_AUTHORIZATION_STATUS_CANDIDATE": b"HEGEL/PHASE2B/STEP5/SIGN/R01_CANDIDATE/PURE/V1\x00",
        "SIGN_R02_SYNTHETIC_BRANCH_CANDIDATE": b"HEGEL/PHASE2B/STEP5/SIGN/R02_CANDIDATE/CUSTODIAN/V1\x00",
        "COUNTERSIGN_R02_SYNTHETIC_BRANCH_CANDIDATE": b"HEGEL/PHASE2B/STEP5/SIGN/R02_CANDIDATE/PURE/V1\x00",
    }
)


class Phase2BStep5AuthorityErrorV1(ValueError):
    """Stable fail-closed error for malformed or unauthenticated artifacts."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BStep5AuthorityErrorV1(code, detail)


def _snapshot(value: object, *, depth: int = 0, counter: list[int] | None = None) -> object:
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > _MAX_NODES or depth > _MAX_DEPTH:
        _fail(FAIL_CANONICAL_V1, "JSON structure exceeds the frozen bounds")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if not -_SAFE_INTEGER <= value <= _SAFE_INTEGER:
            _fail(FAIL_CANONICAL_V1, "integer is outside the exact binary64-safe range")
        return value
    if type(value) is float:
        _fail(FAIL_CANONICAL_V1, "floating-point values are outside Step-5 authority schemas")
    if type(value) is str:
        try:
            encoded = value.encode("utf-8")
        except UnicodeError as exc:
            raise Phase2BStep5AuthorityErrorV1(FAIL_CANONICAL_V1, "invalid Unicode") from exc
        if len(encoded) > _MAX_STRING_UTF8_BYTES:
            _fail(FAIL_CANONICAL_V1, "string exceeds the frozen UTF-8 bound")
        return value
    if type(value) is list:
        return [_snapshot(item, depth=depth + 1, counter=counter) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            _fail(FAIL_CANONICAL_V1, "object keys must be exact strings")
        return {
            key: _snapshot(item, depth=depth + 1, counter=counter)
            for key, item in value.items()
        }
    _fail(FAIL_CANONICAL_V1, "value is outside the exact JSON data model")


def _restore_integral_numbers(value: object) -> object:
    if type(value) is float and value.is_integer() and abs(value) <= _SAFE_INTEGER:
        return int(value)
    if type(value) is list:
        return [_restore_integral_numbers(item) for item in value]
    if type(value) is dict:
        return {key: _restore_integral_numbers(item) for key, item in value.items()}
    return value


def canonical_step5_json_bytes_v1(value: object) -> bytes:
    """Return bounded exact RFC-8785 bytes with no transport newline."""

    try:
        encoded = canonicalize_rfc8785_v1(_snapshot(value))  # type: ignore[arg-type]
    except (RFC8785Error, TypeError, ValueError, RecursionError) as exc:
        raise Phase2BStep5AuthorityErrorV1(FAIL_CANONICAL_V1, str(exc)) from exc
    if not 1 <= len(encoded) <= _MAX_CANONICAL_BYTES or encoded.endswith(b"\n"):
        _fail(FAIL_CANONICAL_V1, "canonical byte size or newline boundary differs")
    return encoded


def decode_canonical_step5_json_v1(
    payload: bytes, *, maximum_bytes: int = _MAX_CANONICAL_BYTES
) -> object:
    """Decode byte-for-byte canonical RFC-8785 under explicit structural bounds."""

    if (
        type(payload) is not bytes
        or type(maximum_bytes) is not int
        or not 1 <= maximum_bytes <= _MAX_CANONICAL_BYTES
        or not 1 <= len(payload) <= maximum_bytes
        or payload.endswith(b"\n")
    ):
        _fail(FAIL_INPUT_V1, "payload must be bounded exact bytes without terminal LF")
    try:
        decoded = decode_and_require_canonical_rfc8785_v1(payload)
    except (RFC8785Error, TypeError, ValueError, RecursionError) as exc:
        raise Phase2BStep5AuthorityErrorV1(FAIL_CANONICAL_V1, str(exc)) from exc
    restored = _snapshot(_restore_integral_numbers(decoded))
    if canonical_step5_json_bytes_v1(restored) != payload:
        _fail(FAIL_CANONICAL_V1, "canonical replay differs after exact number restoration")
    return restored


def phase2b_step5_content_id_v1(kind: str, body: object) -> str:
    """Return a domain-separated SHA-256 ID for one supported body kind."""

    if type(kind) is not str or kind not in _DOMAINS:
        _fail(FAIL_INPUT_V1, "unknown content-ID kind")
    return _PREFIXES[kind] + hashlib.sha256(
        _DOMAINS[kind] + canonical_step5_json_bytes_v1(body)
    ).hexdigest()


def _object(value: object, fields: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        _fail(FAIL_SCHEMA_V1, f"{name} exact field set differs")
    return value


def _text(value: object, name: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > maximum:
        _fail(FAIL_SCHEMA_V1, f"{name} must be a bounded non-empty string")
    return value


def _sha(value: object, name: str) -> str:
    text = _text(value, name)
    if _HEX64_RE.fullmatch(text) is None or text == "0" * 64:
        _fail(FAIL_SCHEMA_V1, f"{name} must be lowercase SHA-256")
    return text


def _identifier(value: object, prefix: str, name: str) -> str:
    text = _text(value, name)
    if not text.startswith(prefix) or _HEX64_RE.fullmatch(text[len(prefix) :]) is None:
        _fail(FAIL_SCHEMA_V1, f"{name} content-ID syntax differs")
    return text


def _generic_identifier(value: object, name: str) -> str:
    text = _text(value, name)
    if _ID_RE.fullmatch(text) is None:
        _fail(FAIL_SCHEMA_V1, f"{name} identifier syntax differs")
    return text


def _image(value: object, name: str) -> str:
    text = _text(value, name)
    if _IMAGE_RE.fullmatch(text) is None or text == "sha256:" + "0" * 64:
        _fail(FAIL_SCHEMA_V1, f"{name} image digest syntax differs")
    return text


def _exact_integer(value: object, expected: int, name: str) -> int:
    if type(value) is not int or value != expected:
        _fail(FAIL_SCHEMA_V1, f"{name} must be exact integer {expected}")
    return value


def _exact_boolean(value: object, expected: bool, name: str) -> bool:
    if type(value) is not bool or value is not expected:
        _fail(FAIL_SCHEMA_V1, f"{name} must be exact boolean {expected}")
    return value


def _exact_equal(left: object, right: object) -> bool:
    # Verified artifacts freeze JSON objects/arrays as MappingProxyType/tuple,
    # while a newly decoded canonical payload necessarily supplies dict/list.
    # Treat only those JSON container representations as equivalent; scalar
    # values remain type-exact so bool/int/float confusion stays fail-closed.
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _exact_equal(item, other)
            for item, other in zip(left, right, strict=True)
        )
    if type(left) is not type(right):
        return False
    return left == right


def _freeze(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


_RESULT_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class Phase2BStep5VerifiedArtifactV1:
    kind: str
    artifact_id: str
    body: Mapping[str, object]
    signature_backends: tuple[str, ...]

    def __new__(cls, token: object = None) -> "Phase2BStep5VerifiedArtifactV1":
        if token is not _RESULT_TOKEN:
            raise TypeError("verified Step-5 artifacts are privately issued")
        return object.__new__(cls)


def _issue(
    kind: str, artifact_id: str, body: dict[str, object], backends: tuple[str, ...] = ()
) -> Phase2BStep5VerifiedArtifactV1:
    result = Phase2BStep5VerifiedArtifactV1(_RESULT_TOKEN)
    object.__setattr__(result, "kind", kind)
    object.__setattr__(result, "artifact_id", artifact_id)
    frozen = _freeze(_snapshot(body))
    assert isinstance(frozen, Mapping)
    object.__setattr__(result, "body", frozen)
    object.__setattr__(result, "signature_backends", backends)
    return result


def _require_verified(
    value: object, kind: str, name: str
) -> Phase2BStep5VerifiedArtifactV1:
    if type(value) is not Phase2BStep5VerifiedArtifactV1 or value.kind != kind:
        _fail(FAIL_INPUT_V1, f"verified {name} required")
    return value


def _validate_stage_b_trust_lock_body(value: object) -> dict[str, object]:
    body = _object(
        value,
        {
            "schema_version",
            "status",
            "canonicalization",
            "authority_profile_id",
            "authority_profile_payload_sha256",
            "signer_registry_id",
            "signer_registry_payload_sha256",
            "qv2_binding",
        },
        "Stage-B trust-lock body",
    )
    if (
        body["schema_version"] != STAGE_B_TRUST_LOCK_SCHEMA_V1
        or body["status"] != "FROZEN_OUT_OF_BAND_STAGE_B_TRUST_LOCK"
        or body["canonicalization"] != "RFC8785_EXACT_BYTES_NO_TRAILING_LF"
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "Stage-B trust-lock header differs")
    _identifier(
        body["authority_profile_id"],
        AUTHORITY_PROFILE_PREFIX_V1,
        "trusted authority profile ID",
    )
    _sha(body["authority_profile_payload_sha256"], "trusted authority profile payload SHA")
    _identifier(
        body["signer_registry_id"],
        SIGNER_REGISTRY_PREFIX_V1,
        "trusted signer registry ID",
    )
    _sha(body["signer_registry_payload_sha256"], "trusted signer registry payload SHA")
    if not _exact_equal(body["qv2_binding"], _EXACT_QV2_BINDING_V1):
        _fail(FAIL_BINDING_V1, "Stage-B trust lock does not bind the exact frozen QV2 /9")
    return body


def load_step5_stage_b_trust_lock_v1(
    payload: bytes,
    *,
    out_of_band_expected_lock_id: str,
    out_of_band_expected_payload_sha256: str,
) -> Phase2BStep5VerifiedArtifactV1:
    """Load one lock only after comparing two independently supplied trust pins.

    The two ``out_of_band_*`` values are a trust input.  They must come from a
    frozen source outside the producer-controlled artifact graph; values copied
    from ``payload`` provide no trust and violate this API's security contract.
    """

    _identifier(
        out_of_band_expected_lock_id,
        STAGE_B_TRUST_LOCK_PREFIX_V1,
        "out-of-band Stage-B trust-lock ID",
    )
    expected_payload_sha = _sha(
        out_of_band_expected_payload_sha256,
        "out-of-band Stage-B trust-lock payload SHA",
    )
    decoded = decode_canonical_step5_json_v1(payload)
    root = _object(decoded, {"body", "id"}, "Stage-B trust lock")
    body = _validate_stage_b_trust_lock_body(root["body"])
    expected_id = phase2b_step5_content_id_v1("stage_b_trust_lock", body)
    if (
        root["id"] != expected_id
        or expected_id != out_of_band_expected_lock_id
        or hashlib.sha256(payload).hexdigest() != expected_payload_sha
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "Stage-B trust lock differs from out-of-band pins")
    return _issue("stage_b_trust_lock", expected_id, body)


def _key_subject(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": row["schema_version"],
        "role": row["role"],
        "control_domain": row["control_domain"],
        "authorized_signature_purposes": row["authorized_signature_purposes"],
        "algorithm": row["algorithm"],
        "key_epoch": row["key_epoch"],
        "public_key_hex": row["public_key_hex"],
    }


def phase2b_step5_role_key_id_v1(row_without_or_with_id: object) -> str:
    row = _object(
        row_without_or_with_id,
        {
            "schema_version",
            "role",
            "control_domain",
            "authorized_signature_purposes",
            "algorithm",
            "key_epoch",
            "public_key_hex",
        }
        | ({"key_id"} if type(row_without_or_with_id) is dict and "key_id" in row_without_or_with_id else set()),
        "role key",
    )
    return phase2b_step5_content_id_v1("role_key", _key_subject(row))


def _validate_key(row: object, expected_role: str) -> dict[str, object]:
    key = _object(
        row,
        {
            "schema_version",
            "role",
            "control_domain",
            "authorized_signature_purposes",
            "algorithm",
            "key_epoch",
            "public_key_hex",
            "key_id",
        },
        "role key",
    )
    if (
        key["schema_version"] != "hegel-machine-phase2b-step5-role-key/1"
        or key["role"] != expected_role
        or key["control_domain"] != ROLE_CONTROL_DOMAINS_V1[expected_role]
        or key["authorized_signature_purposes"] != list(ROLE_SIGNATURE_PURPOSES_V1[expected_role])
        or key["algorithm"] != "Ed25519"
        or type(key["key_epoch"]) is not int
        or key["key_epoch"] != 1
    ):
        _fail(FAIL_SCHEMA_V1, f"{expected_role} key policy differs")
    public_key = _text(key["public_key_hex"], f"{expected_role} public key")
    if len(public_key) != _PUBLIC_KEY_HEX_LENGTH or re.fullmatch(r"[0-9a-f]+", public_key) is None:
        _fail(FAIL_SCHEMA_V1, f"{expected_role} public key encoding differs")
    expected = phase2b_step5_role_key_id_v1(key)
    if key["key_id"] != expected:
        _fail(FAIL_CONTENT_ID_V1, f"{expected_role} key ID differs")
    return key


_OWNER_BINDING_FIELDS: Final = {
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


def _validate_owner_binding(value: object) -> dict[str, object]:
    binding = _object(value, _OWNER_BINDING_FIELDS, "owner authorization binding")
    expected = {
        "authorization_schema": "hegel-machine-phase2b-c1-owner-actual-unsealed-960-authorization/1",
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
    if not _exact_equal(binding, expected):
        _fail(FAIL_BINDING_V1, "owner authorization binding differs from the frozen contract")
    return binding


_THREAT_MODEL: Final = {
    "same_admin_controller": True,
    "owner_accepted_technical_actor_model": True,
    "producer_may_be_malicious": True,
    "producer_outside_authority_roles": True,
    "producer_local_filesystem_claims_authoritative": False,
    "producer_local_filesystem_is_non_authoritative": True,
    "purpose_separated_keys_required": True,
    "distinct_control_domains_required": True,
    "organizational_independence": False,
    "independent_human_actors": False,
    "external_authority": False,
    "remote_attestation": False,
    "hardware_key_nonexportability": False,
    "malicious_admin_resistance": False,
    "malicious_host_root_resistance": False,
    "malicious_container_runtime_resistance": False,
}
_PERMANENT_NONCLAIMS: Final = {
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
    "cas_store_verified": False,
    "logical_time_verified": False,
    "readiness_evidence_overlay_issued": False,
    "technical_control_domains_live_verified": False,
    "organizational_independence": False,
    "independent_human_actors": False,
    "external_authority": False,
    "remote_attestation": False,
    "hardware_key_nonexportability": False,
    "malicious_admin_resistance": False,
}


def _validate_profile_body(value: object) -> dict[str, object]:
    body = _object(
        value,
        {
            "schema_version",
            "status",
            "claim_level",
            "canonicalization",
            "profile_epoch",
            "owner_authorization_binding",
            "readiness_dag_binding",
            "qv2_binding",
            "source_binding",
            "runtime_image_bindings",
            "role_policy",
            "governance_root",
            "threat_model",
            "permanent_nonclaims",
        },
        "authority profile body",
    )
    if (
        body["schema_version"] != AUTHORITY_PROFILE_SCHEMA_V1
        or body["status"] != "FROZEN_OWNER_ACCEPTED_TECHNICAL_AUTHORITY_PROFILE"
        or body["claim_level"]
        != "SAME_ADMIN_TECHNICAL_AUTHORITY_NOT_EXTERNAL_OR_ORGANIZATIONAL_INDEPENDENCE"
        or body["canonicalization"] != "RFC8785_EXACT_BYTES_NO_TRAILING_LF"
        or type(body["profile_epoch"]) is not int
        or body["profile_epoch"] != 1
    ):
        _fail(FAIL_SCHEMA_V1, "authority profile header differs")
    _validate_owner_binding(body["owner_authorization_binding"])
    readiness = _object(body["readiness_dag_binding"], {"schema_version", "readiness_dag_id"}, "readiness binding")
    if readiness != {
        "schema_version": "hegel-machine-phase2b-actual-unsealed-960-readiness-dag/1",
        "readiness_dag_id": READINESS_DAG_ID_V1,
    }:
        _fail(FAIL_BINDING_V1, "readiness DAG binding differs")
    qv2 = _object(
        body["qv2_binding"],
        {
            "schema_version",
            "profile_id",
            "profile_payload_sha256",
            "source_sha256",
            "dependency_lock_sha256",
        },
        "QV2 binding",
    )
    if not _exact_equal(qv2, _EXACT_QV2_BINDING_V1):
        _fail(FAIL_BINDING_V1, "QV2 binding differs from the exact frozen /9 revision")
    source = _object(body["source_binding"], {"source_commit_sha", "source_archive_sha256", "dependency_lock_sha256"}, "source binding")
    if (
        type(source["source_commit_sha"]) is not str
        or _COMMIT_RE.fullmatch(source["source_commit_sha"]) is None
        or source["source_commit_sha"] == "0" * 40
    ):
        _fail(FAIL_SCHEMA_V1, "source commit must be exact lowercase 40-hex")
    _sha(source["source_archive_sha256"], "source archive SHA")
    _sha(source["dependency_lock_sha256"], "dependency lock SHA")
    images = _object(body["runtime_image_bindings"], {"runner", "pure_replayer", "cas_store", "logical_time"}, "runtime images")
    for name, digest in images.items():
        _image(digest, f"{name} image")
    if len(set(images.values())) != len(images):
        _fail(FAIL_SCHEMA_V1, "runtime image digests must be pairwise distinct")
    policy = _object(body["role_policy"], {"roles_in_exact_order", "role_count", "distinct_public_keys", "distinct_key_ids", "distinct_control_domains", "key_epoch"}, "role policy")
    if not _exact_equal(policy, {
        "roles_in_exact_order": list(ROLE_ORDER_V1),
        "role_count": 8,
        "distinct_public_keys": True,
        "distinct_key_ids": True,
        "distinct_control_domains": True,
        "key_epoch": 1,
    }):
        _fail(FAIL_SCHEMA_V1, "exact eight-role policy differs")
    root = _validate_key(body["governance_root"], "GOVERNANCE_ANCHOR")
    if not _exact_equal(body["threat_model"], _THREAT_MODEL):
        _fail(FAIL_CLAIM_BOUNDARY_V1, "threat model overclaims or differs")
    if not _exact_equal(body["permanent_nonclaims"], _PERMANENT_NONCLAIMS):
        _fail(FAIL_CLAIM_BOUNDARY_V1, "permanent nonclaims differ")
    if root["control_domain"] != ROLE_CONTROL_DOMAINS_V1["GOVERNANCE_ANCHOR"]:
        _fail(FAIL_SCHEMA_V1, "governance root control domain differs")
    return body


def _verify_step5_authority_profile_decoded_v1(
    value: object,
    trust_lock: Phase2BStep5VerifiedArtifactV1,
    *,
    payload_sha256: str,
) -> Phase2BStep5VerifiedArtifactV1:
    lock = _require_verified(trust_lock, "stage_b_trust_lock", "Stage-B trust lock")
    root = _object(value, {"body", "id"}, "authority profile")
    body = _validate_profile_body(root["body"])
    expected = phase2b_step5_content_id_v1("authority_profile", body)
    if (
        root["id"] != expected
        or expected != lock.body["authority_profile_id"]
        or payload_sha256 != lock.body["authority_profile_payload_sha256"]
        or not _exact_equal(body["qv2_binding"], lock.body["qv2_binding"])
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "authority profile differs from Stage-B trust lock")
    return _issue("authority_profile", expected, body)


def load_step5_authority_profile_v1(
    payload: bytes, trust_lock: Phase2BStep5VerifiedArtifactV1
) -> Phase2BStep5VerifiedArtifactV1:
    return _verify_step5_authority_profile_decoded_v1(
        decode_canonical_step5_json_v1(payload),
        trust_lock,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


def verify_step5_authority_profile_v1(value: object) -> NoReturn:
    """Fail closed: decoded-object production verification is forbidden."""

    _fail(
        FAIL_TRUST_ANCHOR_V1,
        "use the bytes-only profile loader with an out-of-band Stage-B trust lock",
    )


def phase2b_step5_signature_preimage_v1(
    *, artifact_kind: str, body: object, role: str, purpose: str, signer_key_id: str, key_epoch: int = 1
) -> bytes:
    if purpose not in _SIGNATURE_DOMAINS or role not in ROLE_ORDER_V1 or purpose not in ROLE_SIGNATURE_PURPOSES_V1[role]:
        _fail(FAIL_INPUT_V1, "signature role/purpose pair is not authorized")
    _identifier(signer_key_id, ROLE_KEY_PREFIX_V1, "signer key ID")
    if type(key_epoch) is not int or key_epoch != 1:
        _fail(FAIL_INPUT_V1, "signature key epoch differs")
    subject = {
        "schema_version": "hegel-machine-phase2b-step5-signature-preimage/1",
        "artifact_kind": _text(artifact_kind, "artifact kind"),
        "role": role,
        "purpose": purpose,
        "key_epoch": key_epoch,
        "signer_key_id": signer_key_id,
        "body": body,
    }
    return _SIGNATURE_DOMAINS[purpose] + canonical_step5_json_bytes_v1(subject)


def _validate_signature(value: object, role: str, purpose: str, key: Mapping[str, object]) -> dict[str, object]:
    signature = _object(value, {"schema_version", "algorithm", "role", "purpose", "key_epoch", "signer_key_id", "signature_hex"}, "detached signature")
    if (
        signature["schema_version"] != DETACHED_SIGNATURE_SCHEMA_V1
        or signature["algorithm"] != "Ed25519"
        or signature["role"] != role
        or signature["purpose"] != purpose
        or signature["key_epoch"] != 1
        or signature["signer_key_id"] != key["key_id"]
    ):
        _fail(FAIL_SIGNATURE_V1, "detached signature metadata differs")
    encoded = _text(signature["signature_hex"], "Ed25519 signature")
    if len(encoded) != _SIGNATURE_HEX_LENGTH or re.fullmatch(r"[0-9a-f]+", encoded) is None:
        _fail(FAIL_SIGNATURE_V1, "Ed25519 signature encoding differs")
    return signature


def _verify_signature(
    *,
    artifact_kind: str,
    body: object,
    signature: dict[str, object],
    key: Mapping[str, object],
) -> str:
    preimage = phase2b_step5_signature_preimage_v1(
        artifact_kind=artifact_kind,
        body=body,
        role=signature["role"],
        purpose=signature["purpose"],
        signer_key_id=signature["signer_key_id"],
        key_epoch=signature["key_epoch"],
    )
    if _Ed25519PublicKey is None:
        _fail(FAIL_SIGNATURE_BACKEND_V1, "cryptography Ed25519 backend is unavailable")

    try:
        _Ed25519PublicKey.from_public_bytes(bytes.fromhex(key["public_key_hex"])).verify(
            bytes.fromhex(signature["signature_hex"]), preimage
        )
    except Exception as exc:
        raise Phase2BStep5AuthorityErrorV1(FAIL_SIGNATURE_V1, "Ed25519 signature rejected") from exc
    return "CRYPTOGRAPHY_ED25519"


def _registry_keys(registry: Phase2BStep5VerifiedArtifactV1) -> dict[str, Mapping[str, object]]:
    if type(registry) is not Phase2BStep5VerifiedArtifactV1 or registry.kind != "signer_registry":
        _fail(FAIL_INPUT_V1, "verified signer registry required")
    rows = registry.body["role_keys_in_exact_order"]
    assert type(rows) is tuple
    return {row["role"]: row for row in rows}  # type: ignore[index]


def _require_locked_profile_registry_graph_v1(
    trust_lock: Phase2BStep5VerifiedArtifactV1,
    profile: Phase2BStep5VerifiedArtifactV1,
    registry: Phase2BStep5VerifiedArtifactV1,
) -> tuple[
    Phase2BStep5VerifiedArtifactV1,
    Phase2BStep5VerifiedArtifactV1,
    Phase2BStep5VerifiedArtifactV1,
]:
    """Require one closed lock -> profile -> registry lineage.

    Verified tokens are intentionally reusable within one pure-verifier
    process, so every downstream consumer must re-establish the complete graph
    rather than checking only the two content IDs pinned by a newly supplied
    lock.  In particular, this rejects a hybrid lock that names a profile from
    one valid graph and a registry token previously verified in another.
    """

    lock = _require_verified(trust_lock, "stage_b_trust_lock", "Stage-B trust lock")
    profile = _require_verified(profile, "authority_profile", "authority profile")
    registry = _require_verified(registry, "signer_registry", "signer registry")
    if (
        profile.artifact_id != lock.body["authority_profile_id"]
        or registry.artifact_id != lock.body["signer_registry_id"]
        or registry.body["authority_profile_id"] != profile.artifact_id
        or not _exact_equal(profile.body["qv2_binding"], lock.body["qv2_binding"])
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "Stage-B profile/registry lineage is not closed")
    return lock, profile, registry


def _verify_step5_signer_registry_decoded_v1(
    profile: Phase2BStep5VerifiedArtifactV1,
    trust_lock: Phase2BStep5VerifiedArtifactV1,
    registry_value: object,
    *,
    payload_sha256: str,
) -> Phase2BStep5VerifiedArtifactV1:
    profile = _require_verified(profile, "authority_profile", "authority profile")
    lock = _require_verified(trust_lock, "stage_b_trust_lock", "Stage-B trust lock")
    if profile.artifact_id != lock.body["authority_profile_id"]:
        _fail(FAIL_TRUST_ANCHOR_V1, "profile is not bound by the Stage-B trust lock")
    root = _object(registry_value, {"body", "id", "signatures"}, "signer registry")
    body = _object(root["body"], {"schema_version", "authority_profile_id", "registry_epoch", "status", "revocation_epoch", "role_keys_in_exact_order"}, "signer registry body")
    if (
        body["schema_version"] != SIGNER_REGISTRY_SCHEMA_V1
        or body["authority_profile_id"] != profile.artifact_id
        or type(body["registry_epoch"]) is not int
        or body["registry_epoch"] != 1
        or body["status"] != "ACTIVE"
        or type(body["revocation_epoch"]) is not int
        or body["revocation_epoch"] != 0
        or type(body["role_keys_in_exact_order"]) is not list
        or len(body["role_keys_in_exact_order"]) != 8
    ):
        _fail(FAIL_SCHEMA_V1, "signer registry header or cardinality differs")
    keys = [
        _validate_key(row, role)
        for role, row in zip(ROLE_ORDER_V1, body["role_keys_in_exact_order"], strict=True)
    ]
    if not _exact_equal(keys[0], profile.body["governance_root"]):
        _fail(FAIL_BINDING_V1, "registry governance key differs from profile trust root")
    for field in ("public_key_hex", "key_id", "control_domain"):
        if len({row[field] for row in keys}) != 8:
            _fail(FAIL_SCHEMA_V1, f"registry {field} values are not pairwise distinct")
    expected_id = phase2b_step5_content_id_v1("signer_registry", body)
    if (
        root["id"] != expected_id
        or expected_id != lock.body["signer_registry_id"]
        or payload_sha256 != lock.body["signer_registry_payload_sha256"]
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "signer registry differs from Stage-B trust lock")
    if type(root["signatures"]) is not list or len(root["signatures"]) != 1:
        _fail(FAIL_SIGNATURE_V1, "registry needs exactly one governance signature")
    signature = _validate_signature(root["signatures"][0], "GOVERNANCE_ANCHOR", "SIGN_SIGNER_REGISTRY", keys[0])
    backend = _verify_signature(
        artifact_kind="signer_registry", body=body, signature=signature, key=keys[0]
    )
    return _issue("signer_registry", expected_id, body, (backend,))


def load_step5_signer_registry_v1(
    payload: bytes,
    trust_lock: Phase2BStep5VerifiedArtifactV1,
    profile: Phase2BStep5VerifiedArtifactV1,
) -> Phase2BStep5VerifiedArtifactV1:
    return _verify_step5_signer_registry_decoded_v1(
        profile,
        trust_lock,
        decode_canonical_step5_json_v1(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


def verify_step5_signer_registry_v1(*args: object, **kwargs: object) -> NoReturn:
    """Fail closed: decoded-object or injected-backend verification is forbidden."""

    _fail(
        FAIL_TRUST_ANCHOR_V1,
        "use the bytes-only registry loader with the pinned Stage-B graph",
    )


def _verify_dual_signatures(
    *, artifact_kind: str, body: dict[str, object], signatures: object, keys: Mapping[str, Mapping[str, object]], purposes: tuple[str, str]
) -> tuple[str, str]:
    if type(signatures) is not list or len(signatures) != 2:
        _fail(FAIL_SIGNATURE_V1, f"{artifact_kind} needs exactly two ordered signatures")
    roles = ("SOURCE_RUNTIME_CUSTODIAN", "PURE_REPLAYER")
    backends: list[str] = []
    for raw, role, purpose in zip(signatures, roles, purposes, strict=True):
        signature = _validate_signature(raw, role, purpose, keys[role])
        backends.append(
            _verify_signature(
                artifact_kind=artifact_kind,
                body=body,
                signature=signature,
                key=keys[role],
            )
        )
    return backends[0], backends[1]


_R01_CANDIDATE_CLAIM_BOUNDARY: Final = {
    "owner_authorization_exact_bytes_verified": True,
    "owner_signature_authenticated_by_original_authorization": False,
    "governance_status_ledger_verified": False,
    "latest_revocation_status_verified": False,
    "cas_store_checkpoint_verified": False,
    "logical_time_verified": False,
    "r01_satisfied": False,
    "r02_or_later_satisfied": False,
    "execution_authorized": False,
    "actual_input_consumed": False,
    "organizational_independence": False,
    "independent_human_actors": False,
}


def _verify_owner_authorization_payload_v1(payload: bytes) -> None:
    if type(payload) is not bytes:
        _fail(FAIL_INPUT_V1, "owner authorization payload must be exact bytes")
    if hashlib.sha256(payload).hexdigest() != OWNER_AUTHORIZATION_CONFIG_SHA256_V1:
        _fail(FAIL_BINDING_V1, "owner authorization payload SHA differs")
    decoded = decode_canonical_step5_json_v1(payload)
    wrapper = _object(
        decoded,
        {"authorization_body", "authorization_id"},
        "owner authorization payload",
    )
    if wrapper["authorization_id"] != OWNER_AUTHORIZATION_ID_V1:
        _fail(FAIL_BINDING_V1, "owner authorization payload ID differs")


def _verify_step5_r01_authorization_status_candidate_decoded_v1(
    profile: Phase2BStep5VerifiedArtifactV1,
    registry: Phase2BStep5VerifiedArtifactV1,
    trust_lock: Phase2BStep5VerifiedArtifactV1,
    candidate_value: object,
    owner_authorization_payload: bytes,
) -> Phase2BStep5VerifiedArtifactV1:
    lock, profile, registry = _require_locked_profile_registry_graph_v1(
        trust_lock,
        profile,
        registry,
    )
    _verify_owner_authorization_payload_v1(owner_authorization_payload)
    keys = _registry_keys(registry)
    root = _object(candidate_value, {"body", "id", "signatures"}, "R01 candidate")
    body = _object(
        root["body"],
        {
            "schema_version",
            "authority_profile_id",
            "signer_registry_id",
            "owner_authorization_binding",
            "owner_authorization_payload_sha256",
            "status_checkpoint_candidate",
            "claim_boundary",
        },
        "R01 candidate body",
    )
    if (
        body["schema_version"] != R01_AUTHORIZATION_STATUS_CANDIDATE_SCHEMA_V1
        or body["authority_profile_id"] != profile.artifact_id
        or body["signer_registry_id"] != registry.artifact_id
        or body["owner_authorization_payload_sha256"]
        != OWNER_AUTHORIZATION_CONFIG_SHA256_V1
    ):
        _fail(FAIL_BINDING_V1, "R01 candidate graph binding differs")
    binding = _validate_owner_binding(body["owner_authorization_binding"])
    if not _exact_equal(binding, profile.body["owner_authorization_binding"]):
        _fail(FAIL_BINDING_V1, "R01 candidate owner binding differs from profile")
    checkpoint = _object(
        body["status_checkpoint_candidate"],
        {
            "schema_version",
            "authorization_id",
            "sequence",
            "event",
            "effective_status",
            "previous_status_head_id",
            "revocation_receipt_id",
            "revocation_count",
        },
        "R01 status checkpoint candidate",
    )
    expected_checkpoint = {
        "schema_version": "hegel-machine-phase2b-step5-owner-authorization-status-checkpoint-candidate/1",
        "authorization_id": OWNER_AUTHORIZATION_ID_V1,
        "sequence": 0,
        "event": "INITIAL_ACTIVE_STATUS_CANDIDATE_NOT_LATEST_STATUS_EVIDENCE",
        "effective_status": "ACTIVE_CANDIDATE_NOT_VERIFIED_CURRENT",
        "previous_status_head_id": None,
        "revocation_receipt_id": None,
        "revocation_count": 0,
    }
    if not _exact_equal(checkpoint, expected_checkpoint):
        _fail(FAIL_CLAIM_BOUNDARY_V1, "R01 status candidate differs")
    if not _exact_equal(body["claim_boundary"], _R01_CANDIDATE_CLAIM_BOUNDARY):
        _fail(FAIL_CLAIM_BOUNDARY_V1, "R01 candidate claim boundary differs")
    expected_id = phase2b_step5_content_id_v1("r01_candidate", body)
    if root["id"] != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "R01 candidate ID differs")
    backends = _verify_dual_signatures(
        artifact_kind="r01_candidate",
        body=body,
        signatures=root["signatures"],
        keys=keys,
        purposes=(
            "SIGN_R01_AUTHORIZATION_STATUS_CANDIDATE",
            "COUNTERSIGN_R01_AUTHORIZATION_STATUS_CANDIDATE",
        ),
    )
    return _issue("r01_candidate", expected_id, body, backends)


def load_step5_r01_authorization_status_candidate_v1(
    payload: bytes,
    owner_authorization_payload: bytes,
    trust_lock: Phase2BStep5VerifiedArtifactV1,
    profile: Phase2BStep5VerifiedArtifactV1,
    registry: Phase2BStep5VerifiedArtifactV1,
) -> Phase2BStep5VerifiedArtifactV1:
    return _verify_step5_r01_authorization_status_candidate_decoded_v1(
        profile,
        registry,
        trust_lock,
        decode_canonical_step5_json_v1(payload),
        owner_authorization_payload,
    )


def verify_step5_r01_authorization_status_receipt_v1(
    *args: object, **kwargs: object
) -> NoReturn:
    _fail(
        FAIL_NOT_IMPLEMENTED_V1,
        "R01 evidence receipt is unavailable until governance ledger, CAS, and time exist",
    )


_R02_BRANCH: Final = {
    "selected_branch": "SYNTHETIC_DEVELOPMENT_QUALIFICATION_ONLY",
    "input_class": "SYNTHETIC_DEV_ONLY",
    "synthetic_input": True,
    "actual_corpus_input": False,
    "actual_input_count": 0,
    "actual_campaign_id": None,
    "actual_attempt_id": None,
    "actual_material_consumed": False,
}
_R02_CANDIDATE_CLAIM_BOUNDARY: Final = {
    "declared_binding_equality_verified": True,
    "independent_branch_replay_verified": False,
    "branch_stage_event_verified": False,
    "cas_store_checkpoint_verified": False,
    "logical_time_verified": False,
    "r01_satisfied": False,
    "r02_satisfied": False,
    "r03_satisfied": False,
    "r04_satisfied": False,
    "r05_satisfied": False,
    "actual_execution_authorized": False,
    "actual_corpus_qualified": False,
    "campaign_or_attempt_id_allocated": False,
    "organizational_independence": False,
    "independent_human_actors": False,
}


def _profile_replay_bindings(profile: Phase2BStep5VerifiedArtifactV1) -> dict[str, object]:
    owner = profile.body["owner_authorization_binding"]
    readiness = profile.body["readiness_dag_binding"]
    assert isinstance(owner, Mapping) and isinstance(readiness, Mapping)
    return {
        "owner_authorization_id": owner["authorization_id"],
        "readiness_dag_id": readiness["readiness_dag_id"],
        "protocol_id": owner["protocol_id"],
        "exact_freeze_version": owner["exact_freeze_version"],
        "exact_freeze_id": owner["exact_freeze_id"],
        "authority_class": owner["authority_class"],
        "authority_requirement_id": owner["authority_requirement_id"],
        "actor_profile_content_id": owner["actor_profile_content_id"],
        "actor_profile_canonical_sha256": owner[
            "actor_profile_canonical_sha256"
        ],
        "logical_stage_wire_schema_id": owner["logical_stage_wire_schema_id"],
        "logical_stage_wire_policy_id": owner["logical_stage_wire_policy_id"],
        "stage_event_contract_manifest_id": owner["stage_event_contract_manifest_id"],
    }


def _verify_step5_r02_synthetic_branch_candidate_decoded_v1(
    profile: Phase2BStep5VerifiedArtifactV1,
    registry: Phase2BStep5VerifiedArtifactV1,
    r01_candidate: Phase2BStep5VerifiedArtifactV1,
    trust_lock: Phase2BStep5VerifiedArtifactV1,
    candidate_value: object,
) -> Phase2BStep5VerifiedArtifactV1:
    lock, profile, registry = _require_locked_profile_registry_graph_v1(
        trust_lock,
        profile,
        registry,
    )
    r01 = _require_verified(r01_candidate, "r01_candidate", "R01 candidate")
    if (
        r01.body["authority_profile_id"] != profile.artifact_id
        or r01.body["signer_registry_id"] != registry.artifact_id
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "R02 candidate graph is outside the Stage-B lock")
    keys = _registry_keys(registry)
    root = _object(candidate_value, {"body", "id", "signatures"}, "R02 candidate")
    body = _object(
        root["body"],
        {
            "schema_version",
            "authority_profile_id",
            "signer_registry_id",
            "r01_candidate_id",
            "declared_expected_bindings",
            "declared_observed_selected_branch_bindings",
            "qv2_binding",
            "source_binding",
            "runtime_image_bindings",
            "branch_selection",
            "claim_boundary",
        },
        "R02 candidate body",
    )
    if (
        body["schema_version"] != R02_SYNTHETIC_BRANCH_CANDIDATE_SCHEMA_V1
        or body["authority_profile_id"] != profile.artifact_id
        or body["signer_registry_id"] != registry.artifact_id
        or body["r01_candidate_id"] != r01.artifact_id
    ):
        _fail(FAIL_BINDING_V1, "R02 candidate graph binding differs")
    expected_bindings = _profile_replay_bindings(profile)
    for name in (
        "declared_expected_bindings",
        "declared_observed_selected_branch_bindings",
    ):
        if not _exact_equal(body[name], expected_bindings):
            _fail(FAIL_BINDING_V1, f"R02 candidate {name} differs from frozen bindings")
    for name in ("qv2_binding", "source_binding", "runtime_image_bindings"):
        if not _exact_equal(body[name], profile.body[name]):
            _fail(FAIL_BINDING_V1, f"R02 {name} differs from authority profile")
    if not _exact_equal(body["branch_selection"], _R02_BRANCH):
        _fail(FAIL_CLAIM_BOUNDARY_V1, "R02 branch is not exact synthetic-only/null-actual")
    if not _exact_equal(body["claim_boundary"], _R02_CANDIDATE_CLAIM_BOUNDARY):
        _fail(FAIL_CLAIM_BOUNDARY_V1, "R02 candidate claim boundary differs")
    expected_id = phase2b_step5_content_id_v1("r02_candidate", body)
    if root["id"] != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "R02 candidate ID differs")
    backends = _verify_dual_signatures(
        artifact_kind="r02_candidate",
        body=body,
        signatures=root["signatures"],
        keys=keys,
        purposes=(
            "SIGN_R02_SYNTHETIC_BRANCH_CANDIDATE",
            "COUNTERSIGN_R02_SYNTHETIC_BRANCH_CANDIDATE",
        ),
    )
    return _issue("r02_candidate", expected_id, body, backends)


def load_step5_r02_synthetic_branch_candidate_v1(
    payload: bytes,
    trust_lock: Phase2BStep5VerifiedArtifactV1,
    profile: Phase2BStep5VerifiedArtifactV1,
    registry: Phase2BStep5VerifiedArtifactV1,
    r01_candidate: Phase2BStep5VerifiedArtifactV1,
) -> Phase2BStep5VerifiedArtifactV1:
    return _verify_step5_r02_synthetic_branch_candidate_decoded_v1(
        profile,
        registry,
        r01_candidate,
        trust_lock,
        decode_canonical_step5_json_v1(payload),
    )


def verify_step5_r02_synthetic_branch_receipt_v1(
    *args: object, **kwargs: object
) -> NoReturn:
    _fail(
        FAIL_NOT_IMPLEMENTED_V1,
        "R02 evidence receipt is unavailable until an admitted R01 and independent branch event exist",
    )


def _validate_readiness_dag(value: object) -> None:
    wrapper = _object(value, {"readiness_dag_body", "readiness_dag_id"}, "readiness DAG")
    if wrapper["readiness_dag_id"] != READINESS_DAG_ID_V1:
        _fail(FAIL_BINDING_V1, "readiness DAG ID differs")
    body = _object(
        wrapper["readiness_dag_body"],
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
        },
        "readiness DAG body",
    )
    expected_id = "phase2b_actual_unsealed_960_readiness_dag_v1_" + hashlib.sha256(
        b"HEGEL/PHASE2B/ACTUAL_UNSEALED_960/READINESS_DAG/BODY/V1\x00"
        + canonical_step5_json_bytes_v1(body)
    ).hexdigest()
    if expected_id != READINESS_DAG_ID_V1:
        _fail(FAIL_CONTENT_ID_V1, "readiness DAG body content address differs")
    dag = _object(body["readiness_dag"], {"all_nodes_satisfied", "edge_semantics", "node_count", "nodes_in_canonical_topological_order", "satisfied_node_count"}, "readiness DAG graph")
    nodes = dag["nodes_in_canonical_topological_order"]
    if (
        dag["all_nodes_satisfied"] is not False
        or dag["satisfied_node_count"] != 0
        or dag["node_count"] != 16
        or type(nodes) is not list
        or len(nodes) != 16
    ):
        _fail(FAIL_CLAIM_BOUNDARY_V1, "base readiness DAG must start with zero satisfied nodes")
    for index, node in enumerate(nodes):
        row = _object(node, {"node_id", "dependencies", "evidence_requirement", "initial_satisfied"}, "readiness node")
        if (
            row["node_id"] != READINESS_NODE_ORDER_V1[index]
            or row["dependencies"] != list(READINESS_DEPENDENCIES_V1[index])
            or row["initial_satisfied"] is not False
        ):
            _fail(FAIL_BINDING_V1, "readiness DAG topology differs")


def derive_step5_r01_r02_candidate_snapshot_v1(
    trust_lock: Phase2BStep5VerifiedArtifactV1,
    profile: Phase2BStep5VerifiedArtifactV1,
    registry: Phase2BStep5VerifiedArtifactV1,
    r01_candidate: Phase2BStep5VerifiedArtifactV1,
    r02_candidate: Phase2BStep5VerifiedArtifactV1,
    readiness_dag_payload: bytes,
) -> Phase2BStep5VerifiedArtifactV1:
    """Return a non-evidence snapshot with every readiness node still false."""

    lock, profile, registry = _require_locked_profile_registry_graph_v1(
        trust_lock,
        profile,
        registry,
    )
    r01 = _require_verified(r01_candidate, "r01_candidate", "R01 candidate")
    r02 = _require_verified(r02_candidate, "r02_candidate", "R02 candidate")
    if (
        r01.body["authority_profile_id"] != profile.artifact_id
        or r01.body["signer_registry_id"] != registry.artifact_id
        or r02.body["authority_profile_id"] != profile.artifact_id
        or r02.body["signer_registry_id"] != registry.artifact_id
        or r02.body["r01_candidate_id"] != r01.artifact_id
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "candidate snapshot graph differs from Stage-B lock")
    _validate_readiness_dag(decode_canonical_step5_json_v1(readiness_dag_payload))
    rows = [
        {
            "node_id": node,
            "satisfied": False,
            "candidate_artifact_id": (
                r01.artifact_id if index == 0 else r02.artifact_id if index == 1 else None
            ),
        }
        for index, node in enumerate(READINESS_NODE_ORDER_V1)
    ]
    body = {
        "schema_version": READINESS_CANDIDATE_SNAPSHOT_SCHEMA_V1,
        "claim_level": "CANDIDATE_MECHANICS_ONLY_ALL_READINESS_NODES_FALSE",
        "authority_profile_id": profile.artifact_id,
        "signer_registry_id": registry.artifact_id,
        "readiness_dag_id": READINESS_DAG_ID_V1,
        "node_states_in_canonical_topological_order": rows,
        "satisfied_node_count": 0,
        "highest_contiguous_satisfied_node": None,
        "all_nodes_satisfied": False,
        "actual_campaign_id": None,
        "actual_attempt_id": None,
        "actual_input_consumed": False,
        "r01_satisfied": False,
        "r02_satisfied": False,
        "r03_or_later_unlocked": False,
        "readiness_evidence_overlay_issued": False,
        "cas_store_verified": False,
        "logical_time_verified": False,
    }
    snapshot_id = phase2b_step5_content_id_v1("readiness_candidate", body)
    return _issue("readiness_candidate", snapshot_id, body)


def derive_step5_r01_r02_readiness_overlay_v1(
    *args: object, **kwargs: object
) -> NoReturn:
    """Fail closed until governance ledger, durable CAS, and logical time exist."""

    _fail(
        FAIL_NOT_IMPLEMENTED_V1,
        "readiness evidence overlay issuance is disabled; all sixteen nodes remain false",
    )


__all__ = [
    "ACTOR_PROFILE_CANONICAL_SHA256_V1",
    "ACTOR_PROFILE_CONTENT_ID_V1",
    "AUTHORITY_CLASS_V1",
    "AUTHORITY_PROFILE_PREFIX_V1",
    "AUTHORITY_PROFILE_SCHEMA_V1",
    "AUTHORITY_REQUIREMENT_ID_V1",
    "DETACHED_SIGNATURE_SCHEMA_V1",
    "EXACT_FREEZE_ID_V1",
    "EXACT_FREEZE_VERSION_V1",
    "FAIL_BINDING_V1",
    "FAIL_CANONICAL_V1",
    "FAIL_CLAIM_BOUNDARY_V1",
    "FAIL_CONTENT_ID_V1",
    "FAIL_INPUT_V1",
    "FAIL_NOT_IMPLEMENTED_V1",
    "FAIL_SCHEMA_V1",
    "FAIL_SIGNATURE_BACKEND_V1",
    "FAIL_SIGNATURE_V1",
    "FAIL_TRUST_ANCHOR_V1",
    "LOGICAL_STAGE_WIRE_POLICY_ID_V1",
    "LOGICAL_STAGE_WIRE_SCHEMA_ID_V1",
    "OWNER_AUTHORIZATION_CONFIG_SHA256_V1",
    "OWNER_AUTHORIZATION_ID_V1",
    "OWNER_AUTHORIZATION_SOURCE_SHA256_V1",
    "PROTOCOL_ID_V1",
    "Phase2BStep5AuthorityErrorV1",
    "Phase2BStep5VerifiedArtifactV1",
    "QV2_DEPENDENCY_LOCK_SHA256_V1",
    "QV2_PROFILE_ID_V1",
    "QV2_PROFILE_PAYLOAD_SHA256_V1",
    "QV2_SOURCE_SHA256_V1",
    "R01_AUTHORIZATION_STATUS_CANDIDATE_SCHEMA_V1",
    "R01_CANDIDATE_PREFIX_V1",
    "R02_CANDIDATE_PREFIX_V1",
    "R02_SYNTHETIC_BRANCH_CANDIDATE_SCHEMA_V1",
    "READINESS_CANDIDATE_PREFIX_V1",
    "READINESS_CANDIDATE_SNAPSHOT_SCHEMA_V1",
    "READINESS_DAG_ID_V1",
    "READINESS_NODE_ORDER_V1",
    "ROLE_CONTROL_DOMAINS_V1",
    "ROLE_KEY_PREFIX_V1",
    "ROLE_ORDER_V1",
    "ROLE_SIGNATURE_PURPOSES_V1",
    "SIGNER_REGISTRY_PREFIX_V1",
    "SIGNER_REGISTRY_SCHEMA_V1",
    "STAGE_B_TRUST_LOCK_PREFIX_V1",
    "STAGE_B_TRUST_LOCK_SCHEMA_V1",
    "STAGE_EVENT_CONTRACT_MANIFEST_ID_V1",
    "canonical_step5_json_bytes_v1",
    "decode_canonical_step5_json_v1",
    "derive_step5_r01_r02_candidate_snapshot_v1",
    "derive_step5_r01_r02_readiness_overlay_v1",
    "load_step5_authority_profile_v1",
    "load_step5_r01_authorization_status_candidate_v1",
    "load_step5_r02_synthetic_branch_candidate_v1",
    "load_step5_signer_registry_v1",
    "load_step5_stage_b_trust_lock_v1",
    "phase2b_step5_content_id_v1",
    "phase2b_step5_role_key_id_v1",
    "phase2b_step5_signature_preimage_v1",
    "verify_step5_authority_profile_v1",
    "verify_step5_r01_authorization_status_receipt_v1",
    "verify_step5_r02_synthetic_branch_receipt_v1",
    "verify_step5_signer_registry_v1",
]
